"""Run biological sensitivity and structure metadata stages together."""

from __future__ import annotations

from add_structure_metadata_layer import main as structure_main
from analyze_biological_sensitivity import main as sensitivity_main


def main() -> int:
    sensitivity_code = sensitivity_main()
    structure_code = structure_main()
    return sensitivity_code or structure_code


if __name__ == "__main__":
    raise SystemExit(main())
