#!/bin/bash
set -euo pipefail

echo "run_planrec_dynamic_planner.sh is a compatibility wrapper; use run_planrec_opal_mask_router.sh."
exec bash ./run_planrec_opal_mask_router.sh
