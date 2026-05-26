#!/bin/bash
set -euo pipefail

echo "run_planrec_oracle_templates.sh is a compatibility wrapper; use run_planrec_oracle_masks.sh."
exec bash ./run_planrec_oracle_masks.sh
