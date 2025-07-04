# Bug Ninja

OpenHands is used in [BugNinja](https://github.com/apps/bugninjaai). With some customization modifications.

## Development
The current version is based on OpanHands 0.46.0 (commit hash `76914e3c26cc4035c04f5ab3d31de6a0a4d5d293`)

## Deployment
Docker is used to deploy OpenHands to BugNinja server.

To build and push docker image:
```bash
docker build --platform linux/amd64 -t danielwpz/openhands:{version}-amd64 -f containers/app/Dockerfile .
docker push danielwpz/openhands:{version}-amd64
```

Then, on the server, pull the docker image:
```bash
docker pull danielwpz/openhands:{version}-amd64
```
