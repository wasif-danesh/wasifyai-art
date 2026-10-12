"""ComfyUI API-format workflows and their map files (same format as wasifyai).

Only the map file knows node IDs; Python names inputs, never nodes."""

import copy
import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class Target(BaseModel):
    node: str
    field: str


class Source(BaseModel):
    node: str
    output: int = 0


class OutputSpec(BaseModel):
    node: str
    type: str = "image"


class LoraChain(BaseModel):
    """Where the model flows: loader nodes are inserted from `from_` to every `to`."""

    loader: str = "LoraLoaderModelOnly"
    max: int = Field(default=5, ge=1, le=10)
    from_: Source = Field(alias="from")
    to: list[Target]
    model_config = {"populate_by_name": True}


class WorkflowMap(BaseModel):
    workflow: str
    capability: str
    families: list[str] = Field(default_factory=list)
    licence: str | None = None
    models: list[str] = Field(default_factory=list)
    inputs: dict[str, Target | list[Target]]
    outputs: dict[str, OutputSpec] = Field(default_factory=dict)
    lora_chain: LoraChain | None = None

    def targets(self, name: str) -> list[Target]:
        v = self.inputs[name]
        return v if isinstance(v, list) else [v]


class Workflow:
    def __init__(self, name: str, graph: dict[str, Any], mapping: WorkflowMap):
        self.name = name
        self.graph = graph
        self.map = mapping

    def problems(self) -> list[str]:
        out = []
        for name in self.map.inputs:
            for t in self.map.targets(name):
                if t.node not in self.graph:
                    out.append(f"{self.name}: input {name!r} maps to missing node {t.node!r}")
        chain = self.map.lora_chain
        if chain is not None:
            if chain.from_.node not in self.graph:
                out.append(f"{self.name}: lora_chain starts at missing node {chain.from_.node!r}")
            for t in chain.to:
                node = self.graph.get(t.node)
                if node is None:
                    out.append(f"{self.name}: lora_chain targets missing node {t.node!r}")
                elif node.get("inputs", {}).get(t.field) != [chain.from_.node, chain.from_.output]:
                    out.append(
                        f"{self.name}: lora_chain target {t.node}.{t.field} does not "
                        f"take the model from node {chain.from_.node!r}"
                    )
        for name, o in self.map.outputs.items():
            if o.node not in self.graph:
                out.append(f"{self.name}: output {name!r} maps to missing node {o.node!r}")
        return out

    def fill(
        self, values: dict[str, Any], loras: list[tuple[str, float]] | None = None
    ) -> dict[str, Any]:
        """The graph with inputs set and, in order, one loader node per LoRA spliced into the
        model's path (from -> lora 1 -> ... -> lora n -> every `to`)."""
        unknown = set(values) - set(self.map.inputs)
        if unknown:
            raise ValueError(f"workflow {self.name!r} has no input(s) {sorted(unknown)}")
        graph = copy.deepcopy(self.graph)
        for name, value in values.items():
            for t in self.map.targets(name):
                graph[t.node].setdefault("inputs", {})[t.field] = value
        if loras:
            chain = self.map.lora_chain
            if chain is None:
                raise ValueError(f"workflow {self.name!r} does not accept LoRAs")
            if len(loras) > chain.max:
                raise ValueError(f"workflow {self.name!r} accepts at most {chain.max} LoRAs")
            next_id = max(int(k) for k in graph if k.isdigit()) + 1
            prev: list[Any] = [chain.from_.node, chain.from_.output]
            for file, weight in loras:
                node_id = str(next_id)
                next_id += 1
                graph[node_id] = {
                    "class_type": chain.loader,
                    "inputs": {"model": prev, "lora_name": file, "strength_model": weight},
                }
                prev = [node_id, 0]
            for t in chain.to:
                graph[t.node]["inputs"][t.field] = prev
        return graph


class WorkflowRegistry:
    def __init__(self, workflows: list[Workflow]):
        self._by_name = {w.name: w for w in workflows}

    @classmethod
    def load(cls, directory: Path) -> "WorkflowRegistry":
        found = []
        for map_path in sorted(directory.glob("*.map.yaml")):
            mapping = WorkflowMap.model_validate(yaml.safe_load(map_path.read_text()))
            graph = json.loads((directory / mapping.workflow).read_text())
            found.append(Workflow(map_path.name.removesuffix(".map.yaml"), graph, mapping))
        return cls(found)

    def problems(self) -> list[str]:
        return [p for w in self._by_name.values() for p in w.problems()]

    def get(self, name: str) -> Workflow | None:
        return self._by_name.get(name)

    def find(self, capability: str, family: str) -> Workflow | None:
        for w in self._by_name.values():
            if w.map.capability == capability and family in w.map.families:
                return w
        return None
