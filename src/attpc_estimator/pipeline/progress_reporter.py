from dataclasses import dataclass
from tqdm import tqdm


class ProgressReporter:
	def __init__(self, description: str) -> None:
		self.description = description

	def report_start(self, *, message: str) -> None:
		raise NotImplementedError

	def report_progress(self, current: int, *, message: str = "") -> None:
		raise NotImplementedError

	def report_finish(self, *, message: str) -> None:
		raise NotImplementedError


class TqdmProgressReporter(ProgressReporter):
	def __init__(self, description: str) -> None:
		super().__init__(description)
		self._bar: tqdm | None = None
		self._current = 0

	def report_start(self, *, message: str = "") -> None:
		self.report_finish()
		self._current = 0
		self._bar = tqdm(total=100, desc=self.description)

	def report_progress(self, current: int, *, message: str = "") -> None:
		if self._bar is None:
			return
		bounded = max(int(current), 0)
		delta = max(0, bounded - self._current)
		if delta:
			self._bar.update(delta)
		self._current = max(self._current, bounded)
		if message:
			self._bar.set_postfix_str(message)

	def report_finish(self, *, message: str = "") -> None:
		if self._bar is not None:
			self._bar.close()
			self._bar = None
		self._current = 0