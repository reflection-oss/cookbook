# Contributing

## Adding a recipe

1. Create a folder for the recipe in the right place under `recipes/`.
2. Write a README with every section in [README sections](#readme-sections). Do not delete sections.
3. Keep dependencies and sample data inside the recipe folder. Pin versions.
4. Add a `smoke_test.py` (or `smoke_test.sh`) that runs in under a few minutes and checks the expected output.
5. Run the recipe end to end on real hardware and record what you used in the README.
6. Open a pull request using the pull request template.

## README sections

Start with the recipe title and one sentence on what it does, then these sections in order:

- **What you'll build**: the concrete outcome, what exists at the end that didn't before.
- **What you'll learn**: the takeaway you can apply elsewhere.
- **What you need**: pinned model version, hardware tested, dependencies (`requirements.txt`, pinned), and environment variables (see `.env.example`).
- **Run it**: install and run commands, the real expected output, and the smoke test command.
- **Common fixes**: symptoms and their fixes.
- **Next steps**: how to adapt or extend the example, with links to related recipes and docs.

## A complete recipe has

- A short README with every section above
- Pinned dependencies and model version
- Runnable scripts or a notebook
- Public or synthetic input with its source noted
- Expected output
- A smoke test

Training recipes also report the task metric before and after, the hardware and runtime measured, and how to run the trained result.

## Rules

- Do not commit private code, private data, internal strategy material, or secrets. Use `.env.example` for any variables.
- Do not create a folder for a partner or framework before the integration is available and tested.
- Scripts for straightforward workflows, notebooks only when step-by-step explanation helps.

## Using an AI assistant

`skills/community-contributions/SKILL.md` packages this guide as a skill. Load it in your assistant and it will help you choose a recipe, write the README sections, test it, and prepare the pull request.
