# Release Checklist

## Repository Scope

- [ ] Keep the repository focused on reproducibility artifacts only
- [ ] Do not upload manuscript snapshots unless intentionally decided
- [ ] Do not upload paper-facing result tables unless intentionally decided
- [ ] Do not upload private or institution-derived clinical data

## Code and Configs

- [ ] Scripts run from the released directory structure
- [ ] Config paths do not depend on unpublished local folders
- [ ] Requirements are up to date
- [ ] Public corpus reconstruction scripts are present

## Data

- [ ] Keyword manifest is present
- [ ] Processed public benchmark splits are present
- [ ] Upstream public sources are documented
- [ ] No large mirrored public corpora are accidentally committed

## Documentation

- [ ] README reflects what is actually released
- [ ] Data README reflects what is actually released
- [ ] Results README reflects what is actually released
- [ ] Citation metadata is current
- [ ] Data-availability statement in the manuscript matches the repository scope

## Final Sanity Checks

- [ ] `git status` is clean before pushing
- [ ] No passwords, API keys, or tokens are present
- [ ] No remote checkpoints or transient server logs are present
- [ ] Public GitHub page renders README correctly
