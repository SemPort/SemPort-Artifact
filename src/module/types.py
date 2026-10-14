from __future__ import annotations

from dataclasses import dataclass, field, fields
from enum import Enum

class AnchorStatus(str, Enum):

    EXACT = 'EXACT'
    FUZZY = 'FUZZY'
    STRUCTURAL = 'STRUCTURAL'
    CROSS_FILE = 'CROSS_FILE'
    MERGED = 'MERGED'
    REHOME = 'REHOME'
    FILE_SCOPE = 'FILE_SCOPE'
    UNRESOLVED = 'UNRESOLVED'

def _known_extra(d: dict, names: set[str]) -> tuple[dict, dict]:
    known = {k: v for k, v in d.items() if k in names}
    return known, {k: v for k, v in d.items() if k not in names}

def _clean(d: dict) -> dict:
    return {k: v for k, v in d.items() if v is not None}

def _field_names(cls) -> set[str]:
    return {f.name for f in fields(cls)}

@dataclass
class Hunk:

    delete_lines: list = field(default_factory=list)
    delete_lineno: list[int] = field(default_factory=list)
    add_lines: list = field(default_factory=list)
    add_lineno: list[int] = field(default_factory=list)
    context_before: list[str] = field(default_factory=list)
    context_after: list[str] = field(default_factory=list)
    signature: str | None = None
    old_start: int | None = None
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> 'Hunk':
        known, extra = _known_extra(d, _field_names(cls))
        return cls(**known, extra=extra)

    def to_dict(self) -> dict:
        d = _clean({f.name: getattr(self, f.name)
                    for f in fields(self) if f.name != 'extra'})
        d.update(self.extra)
        return d

@dataclass
class Unit:

    unit_id: str
    file: str = ''
    func: str = ''
    hunks: list[Hunk] = field(default_factory=list)
    raw_diff: str | None = None
    pc_file: str | None = None
    pc_file_src: str | None = None
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> 'Unit':
        d = dict(d)
        hunks = d.pop('hunks', None)
        known, extra = _known_extra(d, _field_names(cls))
        known['hunks'] = [Hunk.from_dict(h) for h in (hunks or [])]
        return cls(**known, extra=extra)

    def to_dict(self) -> dict:
        d = _clean({f.name: getattr(self, f.name)
                    for f in fields(self)
                    if f.name not in ('extra', 'hunks')})
        d['hunks'] = [h.to_dict() for h in self.hunks]
        d.update(self.extra)
        return d

    @property
    def target_file(self) -> str:
        return self.pc_file or self.file

@dataclass
class Intent:

    summary: str | None = None
    security_goal: str | None = None
    behavioral_change: str | None = None
    structural_change: str | None = None
    hunk_role: str | None = None
    role_evidence: str | None = None
    required: list[str] = field(default_factory=list)
    conditional: list[dict] = field(default_factory=list)
    invariants: list[str] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> 'Intent':
        known, extra = _known_extra(d, _field_names(cls))
        return cls(**known, extra=extra)

    def to_dict(self) -> dict:
        d = _clean({f.name: getattr(self, f.name)
                    for f in fields(self) if f.name != 'extra'})
        d.update(self.extra)
        return d

@dataclass
class Candidate:

    file: str = ''
    name: str = ''
    start_line: int = 0
    end_line: int = 0
    score: float | None = None
    signals: dict | None = None
    params: int | None = None
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> 'Candidate':
        known, extra = _known_extra(d, _field_names(cls))
        return cls(**known, extra=extra)

    def to_dict(self) -> dict:
        d = _clean({f.name: getattr(self, f.name)
                    for f in fields(self) if f.name != 'extra'})
        d.update(self.extra)
        return d

@dataclass
class Anchor:

    unit_id: str = ''
    file: str = ''
    func: str = ''
    status: AnchorStatus | str = AnchorStatus.UNRESOLVED
    confidence: float = 0.0
    chosen: Candidate | None = None
    candidates: list[Candidate] = field(default_factory=list)
    pa_func: str | None = None
    merge: dict | None = None
    rehome: dict | None = None
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> 'Anchor':
        d = dict(d)
        chosen = d.pop('chosen', None)
        candidates = d.pop('candidates', None) or []
        known, extra = _known_extra(d, _field_names(cls))
        if chosen is not None:
            known['chosen'] = Candidate.from_dict(chosen)
        known['candidates'] = [Candidate.from_dict(c) for c in candidates]
        raw = known.get('status')
        if raw is None:
            known['status'] = AnchorStatus.UNRESOLVED
        else:
            try:
                known['status'] = AnchorStatus(raw)
            except ValueError:
                pass
        return cls(**known, extra=extra)

    def to_dict(self) -> dict:
        d = _clean({'unit_id': self.unit_id, 'file': self.file,
                    'func': self.func, 'status': self.status_name,
                    'confidence': self.confidence,
                    'pa_func': self.pa_func,
                    'merge': self.merge, 'rehome': self.rehome,
                    'chosen': self.chosen.to_dict() if self.chosen else None,
                    'candidates': [c.to_dict() for c in self.candidates]})
        d.update(self.extra)
        return d

    @property
    def status_name(self) -> str:
        return (self.status.value if isinstance(self.status, AnchorStatus)
                else str(self.status))

    @property
    def is_file_scope(self) -> bool:
        return self.status_name == 'FILE_SCOPE'

@dataclass
class Telemetry:

    prompt_tokens: int = 0
    completion_tokens: int = 0
    llm_ms: float = 0.0
    attempts: int = 0
    wall_ms: float = 0.0

@dataclass
class PatchGroup:

    key: tuple = ()
    file: str = ''
    anchor: Anchor | None = None
    units: list[Unit] = field(default_factory=list)
    safe_id: str = ''
    func_text: str | None = None

    @property
    def unit_ids(self) -> list[str]:
        return [u.unit_id for u in self.units]

    def to_legacy_dict(self) -> dict:
        return {'key': self.key, 'file': self.file,
                'anchor': self.anchor.to_dict() if self.anchor else None,
                'units': [u.to_dict() for u in self.units],
                'safe_id': self.safe_id, 'func_text': self.func_text}

@dataclass
class PatchedFunction:

    group_id: str = ''
    unit_ids: list[str] = field(default_factory=list)
    file: str = ''
    funcs: list[str] = field(default_factory=list)
    anchor_status: str = ''
    llm_ok: bool = False
    parse_ok: bool = False
    error: str = ''
    empty_retries: int = 0
    validator_issues: list[str] = field(default_factory=list)
    validator_retries: int = 0
    memory_retries: int = 0
    dispositions: list[dict] = field(default_factory=list)
    unit_regressions: list[dict] = field(default_factory=list)
    mechanical: bool = False
    direct_apply: bool = False
    noop_with_changes: bool = False
    text: str | None = None
    telemetry: Telemetry = field(default_factory=Telemetry)
    splice: str | None = None
    unit_apply_ok: bool | None = None
    unit_apply_msg: str | None = None

    def to_record(self) -> dict:
        r: dict = {'group_id': self.group_id,
                   'unit_ids': list(self.unit_ids),
                   'file': self.file,
                   'funcs': list(self.funcs),
                   'anchor_status': self.anchor_status,
                   'llm_ok': self.llm_ok,
                   'parse_ok': self.parse_ok,
                   'error': self.error,
                   'empty_retries': self.empty_retries,
                   'validator_issues': list(self.validator_issues),
                   'dispositions': list(self.dispositions),
                   'unit_regressions': list(self.unit_regressions),
                   'validator_retries': self.validator_retries,
                   'memory_retries': self.memory_retries,
                   'prompt_tokens': self.telemetry.prompt_tokens,
                   'completion_tokens': self.telemetry.completion_tokens,
                   'llm_ms': self.telemetry.llm_ms,
                   'attempts': self.telemetry.attempts,
                   'wall_ms': self.telemetry.wall_ms}
        if self.mechanical:
            r['mechanical'] = True
        if self.direct_apply:
            r['direct_apply'] = True
        if self.noop_with_changes:
            r['noop_with_changes'] = True
        if self.splice is not None:
            r['splice'] = self.splice
        if self.unit_apply_ok is not None:
            r['unit_apply_ok'] = self.unit_apply_ok
        if self.unit_apply_msg is not None:
            r['unit_apply_msg'] = self.unit_apply_msg
        return r
