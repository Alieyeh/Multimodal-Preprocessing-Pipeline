from __future__ import annotations

from dataclasses import dataclass

from mmprep.common.console import colorize, status_text


@dataclass
class ProgressPrinter:
    """Small human-readable CLI progress helper.

    Inputs:
    - total_steps: number of coarse pipeline stages.

    Outputs:
    - Console progress messages with stage indices.
    """
    total_steps: int
    current_step: int = 0

    def stage(self, message: str) -> None:
        self.current_step += 1
        print(f"{colorize(f'[{self.current_step}/{self.total_steps}]', 'blue', 'bold')} {status_text('STAGE', 'stage')} {message}", flush=True)

    def item(self, index: int, total: int, message: str) -> None:
        print(f"    {colorize(f'[{index}/{total}]', 'cyan')} {message}", flush=True)
