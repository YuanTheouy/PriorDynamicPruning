#!/bin/bash
set -euo pipefail

echo "run_planrec_static_templates.sh is a compatibility wrapper; use run_planrec_static_masks.sh."
exec bash ./run_planrec_static_masks.sh
