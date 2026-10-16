"""Scene dependency graph (DAG) — Item #11.

Builds a DAG of scenes so the agent knows which scenes can be generated
in parallel and which must wait for shared assets. Scenes that depend
on the same shared asset are grouped; scenes with no dependencies
can run in any order.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SceneNode:
    """A node in the scene dependency graph."""

    scene_id: str
    description: str
    dependencies: set[str] = field(default_factory=set)  # other scene_ids or asset names
    shared_asset_deps: set[str] = field(default_factory=set)  # e.g. {"logo", "background"}
    parallel_group: int = 0  # scenes in the same group can run in parallel

    def to_dict(self) -> dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "description": self.description,
            "dependencies": sorted(self.dependencies),
            "shared_asset_deps": sorted(self.shared_asset_deps),
            "parallel_group": self.parallel_group,
        }


@dataclass
class SceneDependencyGraph:
    """A DAG of scene dependencies."""

    nodes: dict[str, SceneNode] = field(default_factory=dict)
    parallel_groups: list[list[str]] = field(default_factory=list)

    def add_scene(
        self,
        scene_id: str,
        description: str,
        *,
        depends_on: set[str] | None = None,
        shared_assets: set[str] | None = None,
    ) -> SceneNode:
        """Add a scene to the graph."""
        node = SceneNode(
            scene_id=scene_id,
            description=description,
            dependencies=depends_on or set(),
            shared_asset_deps=shared_assets or set(),
        )
        self.nodes[scene_id] = node
        return node

    def topological_sort(self) -> list[str]:
        """Return scenes in dependency order (dependencies first)."""
        visited: set[str] = set()
        result: list[str] = []
        for node_id in self.nodes:
            if node_id not in visited:
                self._visit(node_id, visited, result)
        return result

    def _visit(self, node_id: str, visited: set[str], result: list[str]) -> None:
        if node_id in visited:
            return
        visited.add(node_id)
        node = self.nodes.get(node_id)
        if node:
            for dep in node.dependencies:
                if dep in self.nodes:
                    self._visit(dep, visited, result)
            result.append(node_id)

    def compute_parallel_groups(self) -> list[list[str]]:
        """Group scenes that can run in parallel (no inter-dependencies).

        Scenes in the same group share no dependencies with each other
        and can be generated concurrently.
        """
        # Reset groups
        for node in self.nodes.values():
            node.parallel_group = 0

        groups: list[list[str]] = []
        assigned: set[str] = set()
        order = self.topological_sort()

        for scene_id in order:
            if scene_id in assigned:
                continue
            node = self.nodes[scene_id]
            # Check if all dependencies are already assigned
            unmet_deps = node.dependencies - assigned
            if unmet_deps:
                # Dependencies not yet met — defer to next iteration
                continue
            # Start a new group with this scene + any unassigned scenes
            # that have no unmet deps and share no dependencies with this one
            group = [scene_id]
            assigned.add(scene_id)
            node.parallel_group = len(groups)
            for other_id in order:
                if other_id in assigned:
                    continue
                other = self.nodes[other_id]
                other_unmet = other.dependencies - assigned
                if other_unmet:
                    continue
                # No unmet deps — can join this group if it doesn't depend on
                # scenes already in the group
                if not (other.dependencies & set(group)):
                    group.append(other_id)
                    assigned.add(other_id)
                    other.parallel_group = len(groups)
            groups.append(group)

        self.parallel_groups = groups
        return groups

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": {sid: n.to_dict() for sid, n in self.nodes.items()},
            "parallel_groups": self.parallel_groups,
            "topological_order": self.topological_sort(),
        }


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

_SHARED_ASSET_RE = re.compile(
    r"\b(logo|background|font|color|palette|easing|gradient|intro|outro)\b",
    re.IGNORECASE,
)


def build_dependency_graph(
    scene_descriptions: list[tuple[str, str]],
    *,
    shared_assets: set[str] | None = None,
) -> SceneDependencyGraph:
    """Build a dependency graph from scene descriptions.

    Args:
        scene_descriptions: List of (scene_id, description) tuples.
        shared_assets: Set of shared asset names that scenes may depend on.
            If None, auto-detected from descriptions.

    Returns:
        A SceneDependencyGraph with parallel groups computed.
    """
    graph = SceneDependencyGraph()
    if shared_assets is None:
        shared_assets = {"color_palette", "font_loader", "easing"}  # defaults

    for scene_id, description in scene_descriptions:
        # Detect which shared assets this scene needs
        detected = set()
        lowered = description.lower()
        for asset in shared_assets:
            asset_keyword = asset.split("_")[0]  # "color_palette" → "color"
            if asset_keyword in lowered:
                detected.add(asset)
        graph.add_scene(
            scene_id=scene_id,
            description=description,
            shared_assets=detected,
        )

    graph.compute_parallel_groups()
    return graph
