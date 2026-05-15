# Security Policy

## Scope

This is a research-methodology project. Runtime is stdlib-only, with no network code paths, no user-data handling, and no privileged operations. The attack surface is correspondingly small.

In scope:

- Code-execution flaws in `wmel.benchmark_runner`, the perturbation library, or adapter contracts.
- Malicious-input handling in any code path that consumes JSON reports or external configuration.
- Dependency vulnerabilities flagged by Dependabot (development-only deps; runtime has none).

Out of scope:

- Results being unflattering on a given benchmark.
- Methodology disagreements - those belong in Discussions or in a regular Issue.

## Reporting a vulnerability

If you find a security issue:

1. **Do not open a public Issue or Pull Request.**
2. Contact the maintainer privately - the simplest path is via the email address visible in `git log`.
3. Allow a reasonable disclosure window (30 days) before going public.

A signed CVE is not typically warranted for a project of this scope, but a patch and a release note will land within the disclosure window.

## Supported versions

| Version | Supported |
|---------|-----------|
| 0.5.x   | yes       |
| < 0.5   | no        |

Security updates land on the latest tagged release. Older releases are not back-patched.
