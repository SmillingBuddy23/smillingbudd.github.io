# Threat Model

## Threats
1. Local malware reading process memory.
2. Privilege abuse by background capture process.
3. Model inversion on saved checkpoints.
4. User confusion about surveillance scope.

## Mitigations
- No cloud calls; local-only processing.
- Data minimization: event metadata only, no raw typed characters.
- Optional sensors disabled by default.
- Explicit consent and visible runtime indicators.
- Keep checkpoints local and user-controlled.
