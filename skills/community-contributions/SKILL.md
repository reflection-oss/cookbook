---
name: community-contributions
description: Guide a contributor through adding a recipe to the Reflection cookbook. Use when someone wants to contribute an example, asks which recipe to write, needs help following the recipe README sections, wants to test their recipe, or is preparing a pull request to this repository.
---

# Contributing a recipe

Help the contributor go from idea to a mergeable pull request. Work through the steps in order and do not skip the testing step.

## 1. Choose a recipe

Ask what they want to show and where it fits:

| They want to... | Put it under |
| --- | --- |
| Generate training data | `recipes/customization/data-synthesis/` |
| Fine-tune or post-train | `recipes/customization/sft/` or `rl/` |

Check `CONTRIBUTING.md` and open issues first so they do not duplicate work. Do not create a folder for a provider or framework that is not yet available and tested.

## 2. Scaffold the recipe

Create the new folder and write a README with every section listed in `CONTRIBUTING.md`: **What you'll build**, **What you'll learn**, **What you need**, **Run it**, **Common fixes**, **Next steps**.

A complete recipe folder contains:

- `README.md` with every section
- Runnable code: `main.py` (or `example.py` for Mirror examples), or a notebook only when step-by-step explanation helps
- `requirements.txt` with pinned versions, local to the recipe
- `smoke_test.py` or `smoke_test.sh` that finishes in a few minutes and checks the expected output
- Sample input that is public or synthetic, with its source noted in the README

Training recipes also report the task metric before and after, hardware, runtime, and how to run the trained result. Serving recipes record the image digest, model version, hardware, and chat-template or parser settings.

## 3. Write in the cookbook voice

- Second person, present tense, short sentences.
- Lead with the outcome. Explain a step only when the reader needs it to proceed.
- Paste real output into **Run it**. Never write "output will look something like".
- One concept per recipe. Link to another recipe instead of repeating it.
- Secrets go in environment variables. Point to `.env.example` and never commit a `.env`.

## 4. Test it

Run these before opening the PR and fix anything that fails:

```bash
pip install -r requirements.txt
python main.py           # matches the expected output in the README
python smoke_test.py     # exits 0
python -m compileall -q .
```

Confirm the README states the hardware and versions the run used. If the contributor cannot run the recipe end to end, the PR is not ready.

## 5. Prepare the pull request

- Branch name: `recipe/<area>-<short-name>`
- Title: `Add <area> recipe: <what it does>`
- Fill in every box of `.github/PULL_REQUEST_TEMPLATE.md`. Do not tick a box that is not true.
- Scan the diff for private code, private data, API keys, and internal URLs before pushing.

## Review checklist

Before saying a recipe is ready, confirm each of these:

- [ ] Every README section is filled in
- [ ] Dependencies pinned and local to the recipe
- [ ] Sample data is public or synthetic with source noted
- [ ] Ran end to end; hardware, versions, runtime recorded
- [ ] Smoke test passes
- [ ] No secrets, private code, or private data
