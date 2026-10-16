# Prompt rules appended to TMax's generation prompts

These four blocks are appended to the system prompts of TMax's four generation stages. They keep every environment runnable in a plain container and make the verifiers accept any correct solution. Nothing in this recipe applies them automatically: fix 1 in `TMAX_FIXES.md` at the recipe root names the prompt strings in TMax to append each block to.

## no_tools (all four stages)

IMPORTANT: You are NOT an agent here and you have NO tools. You cannot run bash, inspect any filesystem, or execute anything. Do not emit tool calls or function-call markup of any kind. Answer ONLY with the requested artifact in the format specified above, written from the task description and ground truth alone.

## task (stage 1, task template)

* The environment is a plain Linux container without Docker, systemd, firewall tooling (ufw/nft/iptables) or kernel/network configuration; do not create tasks that need them or any root-only system change. Any service must be startable by a normal user as a plain process on localhost.
* The automated verifier can only test observable behaviour (outputs, files, running services), so specify results, not implementation details.

## initial_test (stage 2, initial-state test)

- Do NOT assert that any process or service is running; the image is tested right after build. Test only files, directories, permissions and file contents.

## verifier (stage 3, final-state test)

Verifier rules (mandatory):
- Test OBSERVABLE BEHAVIOR, never the text of files the agent writes. Run the agent's program or script, call the service, query the database, read the OUTPUT file. Do NOT assert that a source file, script or config contains (or lacks) a substring, function name, import, flag or code pattern. Equivalent implementations must pass.
- Assert only what the task description explicitly specifies. Recompute expected values from the truth; do not invent extra requirements.
- Free text is compared after .strip() and whitespace normalisation, unless the task states the exact bytes. Numbers get a tolerance unless the task states exactness.
- The environment is a plain container and the agent is NOT root: never call or test docker, docker compose, systemctl, systemd, journalctl, ufw, nft/nftables, iptables, modprobe, mount, sysctl, or kernel/network namespaces. Services are plain user-space processes started by the agent; test them over localhost.
