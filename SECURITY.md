# Security Policy

## Supported Versions
| Version | Supported |
| :--- | :--- |
| 0.1.x | Yes |

## Reporting a Vulnerability
If you discover a security vulnerability in Leocherd Iris Energy, please report it via private disclosure rather than opening a public issue.

### File Validation Safeguards
- All incoming video uploads are inspected for valid binary container magic bytes.
- Path traversal sanitization is applied to all file inputs (`os.path.basename` / `Path.resolve()`).
- Upload size limits and frame count caps prevent resource exhaustion Denial of Service (DoS).
- Uploaded assets are never executed as code.
