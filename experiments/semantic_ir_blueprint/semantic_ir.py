"""Strict V0 schema and validation for the Semantic IR."""
from __future__ import annotations

import json
import re
from typing import Annotated, Any, Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator, model_validator


_ID_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class NamedType(StrictModel):
    name: str
    type: str

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        if not _ID_RE.fullmatch(value):
            raise ValueError("name must be an identifier")
        return value


class Definition(StrictModel):
    id: str
    params: list[NamedType]
    type: str
    definition: str
    source_units: list[str]
    source_description: str

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        if not _ID_RE.fullmatch(value):
            raise ValueError("definition id must be an identifier")
        return value

    @model_validator(mode="after")
    def validate_fields(self) -> "Definition":
        if len({param.name for param in self.params}) != len(self.params):
            raise ValueError("definition parameter names must be unique")
        if not self.type.strip() or not self.definition.strip():
            raise ValueError("definition type and body must be non-empty")
        _validate_source_fields(self.source_units, self.source_description)
        return self


class ClaimBase(StrictModel):
    binders: list[NamedType] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_common(self) -> "ClaimBase":
        if len({binder.name for binder in self.binders}) != len(self.binders):
            raise ValueError("claim binder names must be unique")
        if any(not value.strip() for value in self.assumptions):
            raise ValueError("claim assumptions must be non-empty strings")
        return self


class RelationClaim(ClaimBase):
    form: Literal["relation"]
    lhs: str
    relation: str
    rhs: str

    @model_validator(mode="after")
    def validate_relation(self) -> "RelationClaim":
        if not self.lhs.strip() or not self.relation.strip() or not self.rhs.strip():
            raise ValueError("relation claim fields must be non-empty")
        return self


class PredicateClaim(ClaimBase):
    form: Literal["predicate"]
    predicate: str
    arguments: list[str]

    @model_validator(mode="after")
    def validate_predicate(self) -> "PredicateClaim":
        if not self.predicate.strip() or any(not value.strip() for value in self.arguments):
            raise ValueError("predicate and arguments must be non-empty")
        return self


class PropositionClaim(ClaimBase):
    form: Literal["proposition"]
    proposition: str

    @field_validator("proposition")
    @classmethod
    def validate_proposition(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("proposition must be non-empty")
        return value


Claim = Annotated[
    RelationClaim | PredicateClaim | PropositionClaim,
    Field(discriminator="form"),
]


class Node(StrictModel):
    id: str
    kind: Literal["lemma", "theorem"]
    depends_on: list[str]
    claim: Claim
    source_units: list[str]
    source_description: str

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        if not _ID_RE.fullmatch(value):
            raise ValueError("node id must be an identifier")
        return value

    @model_validator(mode="after")
    def validate_fields(self) -> "Node":
        if len(set(self.depends_on)) != len(self.depends_on):
            raise ValueError("node dependencies must be unique")
        _validate_source_fields(self.source_units, self.source_description)
        return self


class SemanticIR(StrictModel):
    definitions: list[Definition]
    nodes: list[Node]

    @model_validator(mode="after")
    def validate_graph(self) -> "SemanticIR":
        definition_ids = [item.id for item in self.definitions]
        node_ids = [item.id for item in self.nodes]
        if len(set(definition_ids)) != len(definition_ids):
            raise ValueError("definition ids must be unique")
        if len(set(node_ids)) != len(node_ids):
            raise ValueError("node ids must be unique")
        if set(definition_ids) & set(node_ids):
            raise ValueError("definition and node ids must be disjoint")
        if not self.nodes:
            raise ValueError("nodes must be non-empty")
        theorem_positions = [i for i, node in enumerate(self.nodes) if node.kind == "theorem"]
        if theorem_positions != [len(self.nodes) - 1]:
            raise ValueError("exactly one theorem is required and it must be the final node")
        earlier: set[str] = set()
        definitions = set(definition_ids)
        for node in self.nodes:
            for dependency in node.depends_on:
                if dependency in definitions:
                    raise ValueError("definitions cannot appear in depends_on")
                if dependency not in earlier:
                    raise ValueError(
                        f"dependency {dependency!r} must refer to an earlier proof node"
                    )
            earlier.add(node.id)
        return self


def _validate_source_fields(source_units: Sequence[str], description: str) -> None:
    if not source_units or len(set(source_units)) != len(source_units):
        raise ValueError("source_units must be non-empty and unique")
    if any(not re.fullmatch(r"S[0-9]{3,}", value) for value in source_units):
        raise ValueError("source_units must contain S-unit identifiers")
    if not description.strip():
        raise ValueError("source_description must be non-empty")


def extract_json_object(content: str) -> dict[str, Any]:
    """Parse a raw JSON object or one fenced JSON object without repair."""
    raw = str(content or "").strip()
    fenced = re.findall(r"```(?:json)?\s*\n(.*?)```", raw, re.DOTALL | re.IGNORECASE)
    candidate = fenced[-1].strip() if fenced else raw
    value = json.loads(candidate)
    if not isinstance(value, dict):
        raise ValueError("Semantic IR response must be a JSON object")
    return value


def parse_semantic_ir(content: str) -> SemanticIR:
    return TypeAdapter(SemanticIR).validate_python(extract_json_object(content), strict=True)


def validate_source_unit_references(ir: SemanticIR, source_unit_ids: Sequence[str]) -> None:
    valid = set(source_unit_ids)
    for kind, values in (("definition", ir.definitions), ("node", ir.nodes)):
        for value in values:
            unknown = [unit for unit in value.source_units if unit not in valid]
            if unknown:
                raise ValueError(f"{kind} {value.id!r} references unknown source units: {unknown}")
