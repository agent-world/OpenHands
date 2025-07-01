import json
import os
import re
from typing import Any, ClassVar

import jinja2

from openhands.core.config import LLMConfig
from openhands.events.event import Event
from openhands.llm.llm import LLM
from openhands.resolver.interfaces.issue import (Issue, IssueHandlerInterface,
                                                 ReviewThread)
from openhands.resolver.utils import extract_image_urls


class ServiceContext:
    issue_type: ClassVar[str]
    default_git_patch: ClassVar[str] = 'No changes made yet'

    def __init__(self, strategy: IssueHandlerInterface, llm_config: LLMConfig | None):
        self._strategy = strategy
        if llm_config is not None:
            self.llm = LLM(llm_config)

    def set_strategy(self, strategy: IssueHandlerInterface) -> None:
        self._strategy = strategy


# Strategy context interface
class ServiceContextPR(ServiceContext):
    issue_type: ClassVar[str] = 'pr'

    def __init__(self, strategy: IssueHandlerInterface, llm_config: LLMConfig):
        super().__init__(strategy, llm_config)

    def _truncate_to_last_n_lines(self, text: str, max_lines: int = 200) -> str:
        """Helper function to truncate text to only the last N lines."""
        if not text:
            return text
        lines = text.split('\n')
        if len(lines) > max_lines:
            return '\n'.join(lines[-max_lines:])
        return text

    def get_clone_url(self) -> str:
        return self._strategy.get_clone_url()

    def download_issues(self) -> list[Any]:
        return self._strategy.download_issues()

    def guess_success(
        self,
        issue: Issue,
        history: list[Event],
        git_patch: str | None = None,
    ) -> tuple[bool, None | list[bool], str]:
        """Guess if the issue is fixed based on the history, issue description and git patch.

        Args:
            issue: The issue to check
            history: The agent's history
            git_patch: Optional git patch showing the changes made
        """
        last_message = history[-1].message

        issues_context = json.dumps(issue.closing_issues, indent=4)
        success_list = []
        explanation_list = []

        # Handle PRs with file-specific review comments
        if issue.review_threads:
            for review_thread in issue.review_threads:
                if issues_context and last_message:
                    success, explanation = self._check_review_thread(
                        review_thread, issues_context, last_message, git_patch
                    )
                else:
                    success, explanation = False, 'Missing context or message'
                success_list.append(success)
                explanation_list.append(explanation)
        # Handle PRs with only thread comments (no file-specific review comments)
        elif issue.thread_comments:
            if issue.thread_comments and issues_context and last_message:
                success, explanation = self._check_thread_comments(
                    issue.thread_comments, issues_context, last_message, git_patch
                )
            else:
                success, explanation = (
                    False,
                    'Missing thread comments, context or message',
                )
            success_list.append(success)
            explanation_list.append(explanation)
        elif issue.review_comments:
            # Handle PRs with only review comments (no file-specific review comments or thread comments)
            if issue.review_comments and issues_context and last_message:
                success, explanation = self._check_review_comments(
                    issue.review_comments, issues_context, last_message, git_patch
                )
            else:
                success, explanation = (
                    False,
                    'Missing review comments, context or message',
                )
            success_list.append(success)
            explanation_list.append(explanation)
        else:
            # No review comments, thread comments, or file-level review comments found
            return False, None, 'No feedback was found to process'

        # Return overall success (all must be true) and explanations
        if not success_list:
            return False, None, 'No feedback was processed'
        return all(success_list), success_list, json.dumps(explanation_list)

    def get_converted_issues(
        self, issue_numbers: list[int] | None = None, comment_id: int | None = None
    ) -> list[Issue]:
        return self._strategy.get_converted_issues(issue_numbers, comment_id)

    def get_instruction(
        self,
        issue: Issue,
        user_instructions_prompt_template: str,
        conversation_instructions_prompt_template: str,
        repo_instruction: str | None = None,
        comment_id: int | None = None,
    ) -> tuple[str, str, list[str]]:
        """Generate instruction for the agent."""
        user_instruction_template = jinja2.Template(user_instructions_prompt_template)
        conversation_instructions_template = jinja2.Template(
            conversation_instructions_prompt_template
        )
        images = []

        issues_str = None
        if issue.closing_issues:
            issues_str = json.dumps(issue.closing_issues, indent=4)
            images.extend(extract_image_urls(issues_str))

                # Handle PRs with review comments (individual review comments, not thread comments)
        review_comments_str = "None"
        focused_review_comment = None

        if comment_id is not None:
            # When a specific comment is requested, show ALL existing review comments for context
            # NOTE: We don't call get_pr_comments here as that's for thread comments, not review comments
            all_review_comments = issue.review_comments or []
            if len(all_review_comments) > 0:
                formatted_comments = []
                for i, comment in enumerate(all_review_comments, 1):
                    if comment and comment.strip():
                        formatted_comments.append(f"<comment_{i}>\n{comment.strip()}\n</comment_{i}>")
                review_comments_str = "\n\n".join(formatted_comments) if formatted_comments else "None"
                images.extend(extract_image_urls("\n".join(all_review_comments)))
            # Use the filtered comment from issue object as the focused one
            if issue.review_comments and len(issue.review_comments) >= 1:
                focused_review_comment = issue.review_comments[0]
        elif issue.review_comments and len(issue.review_comments) > 0:
            formatted_comments = []
            for i, comment in enumerate(issue.review_comments, 1):
                if comment and comment.strip():
                    formatted_comments.append(f"<comment_{i}>\n{comment.strip()}\n</comment_{i}>")
            review_comments_str = "\n\n".join(formatted_comments) if formatted_comments else "None"
            images.extend(extract_image_urls("\n".join(issue.review_comments)))

        # Handle PRs with file-specific review comments
        review_thread_str = "None"
        review_thread_file_str = "None"
        focused_review_thread = None

        if comment_id is not None and issue.review_threads and len(issue.review_threads) > 0:
            # When a specific comment is requested, use the existing filtered threads
            formatted_threads = []
            review_thread_files = []
            for i, review_thread in enumerate(issue.review_threads, 1):
                if review_thread.comment and review_thread.comment.strip():
                    files_info = f" (files: {', '.join(review_thread.files)})" if review_thread.files else ""
                    formatted_threads.append(f"<thread_{i}{files_info}>\n{review_thread.comment.strip()}\n</thread_{i}>")
                    review_thread_files.extend(review_thread.files)
            review_thread_str = "\n\n".join(formatted_threads) if formatted_threads else "None"
            review_thread_file_str = ", ".join(sorted(set(review_thread_files))) if review_thread_files else "None"
            images.extend(extract_image_urls("\n".join([rt.comment for rt in issue.review_threads if rt.comment])))
            # Use the first thread as the focused one (since issue.review_threads contains filtered results)
            if issue.review_threads and issue.review_threads[0].comment:
                focused_review_thread = issue.review_threads[0].comment
        elif issue.review_threads and len(issue.review_threads) > 0:
            formatted_threads = []
            review_thread_files = []
            for i, review_thread in enumerate(issue.review_threads, 1):
                if review_thread.comment and review_thread.comment.strip():
                    files_info = f" (files: {', '.join(review_thread.files)})" if review_thread.files else ""
                    formatted_threads.append(f"<thread_{i}{files_info}>\n{review_thread.comment.strip()}\n</thread_{i}>")
                    review_thread_files.extend(review_thread.files)
            review_thread_str = "\n\n".join(formatted_threads) if formatted_threads else "None"
            review_thread_file_str = ", ".join(sorted(set(review_thread_files))) if review_thread_files else "None"
            images.extend(extract_image_urls("\n".join([rt.comment for rt in issue.review_threads if rt.comment])))

                # Handle PR thread comments (general discussion comments on the PR)
        thread_context = "None"
        focused_thread_comment = None

        if comment_id is not None:
            # When a specific comment is requested, fetch ALL thread comments for context
            all_thread_comments = self._strategy.get_pr_comments(issue.number, comment_id=None) if hasattr(self._strategy, 'get_pr_comments') else None
            if all_thread_comments and len(all_thread_comments) > 0:
                formatted_thread_comments = []
                for i, comment in enumerate(all_thread_comments, 1):
                    if comment and comment.strip():
                        formatted_thread_comments.append(f"<pr_comment_{i}>\n{comment.strip()}\n</pr_comment_{i}>")
                thread_context = "\n\n".join(formatted_thread_comments) if formatted_thread_comments else "None"
                images.extend(extract_image_urls("\n".join(all_thread_comments)))
            # Use the filtered comment from issue object as the focused one
            if issue.thread_comments and len(issue.thread_comments) >= 1:
                focused_thread_comment = issue.thread_comments[0]
        elif issue.thread_comments and len(issue.thread_comments) > 0:
            formatted_thread_comments = []
            for i, comment in enumerate(issue.thread_comments, 1):
                if comment and comment.strip():
                    formatted_thread_comments.append(f"<pr_comment_{i}>\n{comment.strip()}\n</pr_comment_{i}>")
            thread_context = "\n\n".join(formatted_thread_comments) if formatted_thread_comments else "None"
            images.extend(extract_image_urls("\n".join(issue.thread_comments)))

                # Prepare the focused comment to resolve as a separate section
        comment_to_resolve_str = "None"
        if focused_review_comment:
            comment_to_resolve_str = focused_review_comment.strip()
        elif focused_review_thread:
            comment_to_resolve_str = focused_review_thread.strip()
        elif focused_thread_comment:
            comment_to_resolve_str = focused_thread_comment.strip()

        user_instruction = user_instruction_template.render(
            review_comments=review_comments_str or "None",
            review_threads=review_thread_str or "None",
            files=review_thread_file_str or "None",
            thread_context=thread_context or "None",
            comment_to_resolve=comment_to_resolve_str,
        )

        conversation_instructions = conversation_instructions_template.render(
            issues=issues_str, repo_instruction=repo_instruction
        )

        return user_instruction, conversation_instructions, images

    def _check_feedback_with_llm(self, prompt: str) -> tuple[bool, str]:
        """Helper function to check feedback with LLM and parse response."""
        response = self.llm.completion(messages=[{'role': 'user', 'content': prompt}])

        answer = response.choices[0].message.content.strip()
        pattern = r'--- success\n*(true|false)\n*--- explanation*\n((?:.|\n)*)'
        match = re.search(pattern, answer)
        if match:
            return match.group(1).lower() == 'true', match.group(2).strip()
        return False, f'Failed to decode answer from LLM response: {answer}'

    def _check_review_thread(
        self,
        review_thread: ReviewThread,
        issues_context: str,
        last_message: str,
        git_patch: str | None = None,
    ) -> tuple[bool, str]:
        """Check if a review thread's feedback has been addressed."""
        files_context = json.dumps(review_thread.files, indent=4)

        # Truncate all parameters
        issues_context = self._truncate_to_last_n_lines(issues_context)
        feedback = self._truncate_to_last_n_lines(review_thread.comment)
        files_context = self._truncate_to_last_n_lines(files_context)
        last_message = self._truncate_to_last_n_lines(last_message)
        git_patch = self._truncate_to_last_n_lines(git_patch or self.default_git_patch)

        with open(
            os.path.join(
                os.path.dirname(__file__),
                '../prompts/guess_success/pr-feedback-check.jinja',
            ),
            'r',
        ) as f:
            template = jinja2.Template(f.read())

        prompt = template.render(
            issue_context=issues_context,
            feedback=feedback,
            files_context=files_context,
            last_message=last_message,
            git_patch=git_patch,
        )

        return self._check_feedback_with_llm(prompt)

    def _check_thread_comments(
        self,
        thread_comments: list[str],
        issues_context: str,
        last_message: str,
        git_patch: str | None = None,
    ) -> tuple[bool, str]:
        """Check if thread comments feedback has been addressed."""
        thread_context = '\n---\n'.join(thread_comments)

        # Truncate all parameters
        issues_context = self._truncate_to_last_n_lines(issues_context)
        thread_context = self._truncate_to_last_n_lines(thread_context)
        last_message = self._truncate_to_last_n_lines(last_message)
        git_patch = self._truncate_to_last_n_lines(git_patch or self.default_git_patch)

        with open(
            os.path.join(
                os.path.dirname(__file__),
                '../prompts/guess_success/pr-thread-check.jinja',
            ),
            'r',
        ) as f:
            template = jinja2.Template(f.read())

        prompt = template.render(
            issue_context=issues_context,
            thread_context=thread_context,
            last_message=last_message,
            git_patch=git_patch,
        )

        return self._check_feedback_with_llm(prompt)

    def _check_review_comments(
        self,
        review_comments: list[str],
        issues_context: str,
        last_message: str,
        git_patch: str | None = None,
    ) -> tuple[bool, str]:
        """Check if review comments feedback has been addressed."""
        review_context = '\n---\n'.join(review_comments)

        # Truncate all parameters
        issues_context = self._truncate_to_last_n_lines(issues_context)
        review_context = self._truncate_to_last_n_lines(review_context)
        last_message = self._truncate_to_last_n_lines(last_message)
        git_patch = self._truncate_to_last_n_lines(git_patch or self.default_git_patch)

        with open(
            os.path.join(
                os.path.dirname(__file__),
                '../prompts/guess_success/pr-review-check.jinja',
            ),
            'r',
        ) as f:
            template = jinja2.Template(f.read())

        prompt = template.render(
            issue_context=issues_context,
            review_context=review_context,
            last_message=last_message,
            git_patch=git_patch,
        )

        return self._check_feedback_with_llm(prompt)


class ServiceContextIssue(ServiceContext):
    issue_type: ClassVar[str] = 'issue'

    def __init__(self, strategy: IssueHandlerInterface, llm_config: LLMConfig | None):
        super().__init__(strategy, llm_config)

    def _truncate_to_last_n_lines(self, text: str, max_lines: int = 200) -> str:
        """Helper function to truncate text to only the last N lines."""
        if not text:
            return text
        lines = text.split('\n')
        if len(lines) > max_lines:
            return '\n'.join(lines[-max_lines:])
        return text

    def get_base_url(self) -> str:
        return self._strategy.get_base_url()

    def get_branch_url(self, branch_name: str) -> str:
        return self._strategy.get_branch_url(branch_name)

    def get_download_url(self) -> str:
        return self._strategy.get_download_url()

    def get_clone_url(self) -> str:
        return self._strategy.get_clone_url()

    def get_graphql_url(self) -> str:
        return self._strategy.get_graphql_url()

    def get_headers(self) -> dict[str, str]:
        return self._strategy.get_headers()

    def get_authorize_url(self) -> str:
        return self._strategy.get_authorize_url()

    def get_pull_url(self, pr_number: int) -> str:
        return self._strategy.get_pull_url(pr_number)

    def get_compare_url(self, branch_name: str) -> str:
        return self._strategy.get_compare_url(branch_name)

    def download_issues(self) -> list[Any]:
        return self._strategy.download_issues()

    def get_branch_name(
        self,
        base_branch_name: str,
    ) -> str:
        return self._strategy.get_branch_name(base_branch_name)

    def branch_exists(self, branch_name: str) -> bool:
        return self._strategy.branch_exists(branch_name)

    def get_default_branch_name(self) -> str:
        return self._strategy.get_default_branch_name()

    def create_pull_request(self, data: dict[str, Any] | None = None) -> dict[str, Any]:
        if data is None:
            data = {}
        return self._strategy.create_pull_request(data)

    def request_reviewers(self, reviewer: str, pr_number: int) -> None:
        return self._strategy.request_reviewers(reviewer, pr_number)

    def reply_to_comment(self, pr_number: int, comment_id: str, reply: str) -> None:
        return self._strategy.reply_to_comment(pr_number, comment_id, reply)

    def send_comment_msg(self, issue_number: int, msg: str) -> None:
        return self._strategy.send_comment_msg(issue_number, msg)

    def get_issue_comments(
        self, issue_number: int, comment_id: int | None = None
    ) -> list[str] | None:
        return self._strategy.get_issue_comments(issue_number, comment_id)

    def get_instruction(
        self,
        issue: Issue,
        user_instructions_prompt_template: str,
        conversation_instructions_prompt_template: str,
        repo_instruction: str | None = None,
        comment_id: int | None = None,
    ) -> tuple[str, str, list[str]]:
        """Generate instruction for the agent."""
        # Handle issue thread comments
        thread_context = ""
        focused_issue_comment = None

        if comment_id is not None:
            # When a specific comment is requested, fetch ALL issue comments
            all_issue_comments = self._strategy.get_issue_comments(issue.number, comment_id=None) if hasattr(self._strategy, 'get_issue_comments') else None
            if all_issue_comments and len(all_issue_comments) > 0:
                formatted_issue_comments = []
                for i, comment in enumerate(all_issue_comments, 1):
                    if comment and comment.strip():
                        formatted_issue_comments.append(f"<issue_comment_{i}>\n{comment.strip()}\n</issue_comment_{i}>")
                if formatted_issue_comments:
                    thread_context = '\n\n<issue_thread_comments>\n' + "\n\n".join(formatted_issue_comments) + '\n</issue_thread_comments>'
            # Use the filtered comment from issue object as the focused one
            if issue.thread_comments and len(issue.thread_comments) >= 1:
                focused_issue_comment = issue.thread_comments[0]
        elif issue.thread_comments and len(issue.thread_comments) > 0:
            formatted_issue_comments = []
            for i, comment in enumerate(issue.thread_comments, 1):
                if comment and comment.strip():
                    formatted_issue_comments.append(f"<issue_comment_{i}>\n{comment.strip()}\n</issue_comment_{i}>")
            if formatted_issue_comments:
                thread_context = '\n\n<issue_thread_comments>\n' + "\n\n".join(formatted_issue_comments) + '\n</issue_thread_comments>'

        images = []
        images.extend(extract_image_urls(issue.body))
        if thread_context:
            images.extend(extract_image_urls(thread_context))

        # Prepare the main body content for issues
        main_body = issue.title + '\n\n' + issue.body + thread_context

        # Add focused comment section if needed
        if focused_issue_comment:
            main_body += f"\n\n<comment_to_resolve>\n{focused_issue_comment.strip()}\n</comment_to_resolve>"

        user_instructions_template = jinja2.Template(user_instructions_prompt_template)
        user_instructions = user_instructions_template.render(
            body=main_body
        )  # Issue body and comments

        conversation_instructions_template = jinja2.Template(
            conversation_instructions_prompt_template
        )
        conversation_instructions = conversation_instructions_template.render(
            repo_instruction=repo_instruction,
        )

        return user_instructions, conversation_instructions, images

    def guess_success(
        self, issue: Issue, history: list[Event], git_patch: str | None = None
    ) -> tuple[bool, None | list[bool], str]:
        """Guess if the issue is fixed based on the history and the issue description.

        Args:
            issue: The issue to check
            history: The agent's history
            git_patch: Optional git patch showing the changes made
        """
        last_message = history[-1].message
        # Include thread comments in the prompt if they exist
        issue_context = issue.body
        if issue.thread_comments:
            issue_context += '\n\nIssue Thread Comments:\n' + '\n---\n'.join(
                issue.thread_comments
            )

        # Truncate all parameters
        issue_context = self._truncate_to_last_n_lines(issue_context)
        last_message = self._truncate_to_last_n_lines(last_message)
        git_patch = self._truncate_to_last_n_lines(git_patch or self.default_git_patch)

        with open(
            os.path.join(
                os.path.dirname(__file__),
                '../prompts/guess_success/issue-success-check.jinja',
            ),
            'r',
        ) as f:
            template = jinja2.Template(f.read())

        prompt = template.render(
            issue_context=issue_context,
            last_message=last_message,
            git_patch=git_patch,
        )

        print("guess success inputs", prompt)
        response = self.llm.completion(messages=[{'role': 'user', 'content': prompt}])

        answer = response.choices[0].message.content.strip()
        pattern = r'--- success\n*(true|false)\n*--- explanation*\n((?:.|\n)*)'
        match = re.search(pattern, answer)
        if match:
            return match.group(1).lower() == 'true', None, match.group(2)

        return False, None, f'Failed to decode answer from LLM response: {answer}'

    def get_converted_issues(
        self, issue_numbers: list[int] | None = None, comment_id: int | None = None
    ) -> list[Issue]:
        return self._strategy.get_converted_issues(issue_numbers, comment_id)
