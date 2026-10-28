# Contributing

Thanks for taking an interest in Human Sense AI.

## Before making a change

1. Read the project status and limitations in [README.md](README.md).
2. Open an issue first for large changes so the approach can be discussed.
3. Keep changes focused and avoid claiming model accuracy without reproducible evaluation.

## Local setup

Use the Windows setup steps in the README. The application uses a camera and model assets, so runtime checks may require a compatible webcam and the expected files in `models/`.

## Quality checks

Before opening a pull request, run:

```powershell
.\.venv\Scripts\python.exe -m compileall -q src camera_test.py
```

If your change affects camera processing or model inference, also describe what you tested, your Python version, and any relevant dependency versions. Do not include private camera footage or personal data in issues.

## Pull requests

- Explain the problem and the reason for the change.
- List the main files or behaviours changed.
- Include test results and known limitations.
- Add screenshots or a short recording only when they genuinely show the changed behaviour.
- Keep documentation consistent with the implementation.
