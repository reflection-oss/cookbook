# Troubleshooting

- **`docker build` fails with `toomanyrequests`**: keep `ubuntu:22.04` as a local tagged image and never prune it; anonymous Docker Hub pulls are capped at 100 per 6 hours.
- **The disk fills up**: `docker builder prune --filter until=1h` between runs; `docker rmi $(docker images -q 'tmax-task:*')` when the images are no longer needed.
- **Trials die with `EnvironmentStartTimeoutError`**: check that `build_timeout_sec = 3600` reached the staged `task.toml`.
- **Trials die with `NetworkConnectionError` or HTTP 502**: counted as infrastructure failures, not failed attempts, and rerun next time.
