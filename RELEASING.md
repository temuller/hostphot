# Releasing HostPhot

Checklist for a new release (e.g. `X.Y.Z`).

## Before the release

1. Update the version in `src/hostphot/_version.py` and the `## vX.Y.Z` section of `CHANGELOG.md`.
2. Run the tests that do not need the survey archives, and the network tests (or run the "Network tests"
   workflow manually on GitHub):

   ```bash
   pytest -v -m "not network"
   pytest -v -m network -rs
   ```

   Network tests reported as skipped mean that an archive was not available: check that they are not
   skipped for long periods (e.g. a changed service).
3. Merge into `main` and check that the "Tests" workflow passes.

## Build (from a clean checkout)

Local `build/` directories can contain old files (e.g. removed filter files) that end up in the package, so
build from a fresh clone of the release commit:

```bash
git clone https://github.com/temuller/hostphot.git hostphot-release
cd hostphot-release
git checkout vX.Y.Z   # or the release commit
python -m pip install -U build twine
python -m build
twine check dist/*
```

Check the contents of the wheel (e.g. that the filter files of all the surveys are included and no other
files are):

```bash
python -m zipfile -l dist/hostphot-X.Y.Z-py3-none-any.whl | awk '{print $1}' | grep filters/ | awk -F/ '{print $3}' | sort | uniq -c
```

Optionally, install the wheel in a new environment and run the tests against it.

## Publish

1. Upload to PyPI: `twine upload dist/*`.
2. Create the release on GitHub (tag `vX.Y.Z`), with the changelog section as the release notes.
3. Check the documentation on Read the Docs (the `latest` and the new version).
