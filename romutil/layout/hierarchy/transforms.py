"""Hierarchical solve tree transformation passes and composition pipeline."""

from __future__ import annotations

import collections
from typing import Protocol, Sequence, runtime_checkable

from romutil.layout.ir import (
    ArticulationPointCut,
    BridgeCut,
    CutType,
    LayoutGraph,
    SolveTree,
    SolveTreeNode,
    SplitPairCut,
    TreeNodeType,
)
from romutil.layout.simplification.warps import direction_to_displacement
from romutil.models import Direction


@runtime_checkable
class TreeTransformationPass(Protocol):
    """Protocol for hierarchical solve tree transformation passes."""

    def apply(self, tree: SolveTree) -> SolveTree:
        """Apply transformation pass to a solve tree and return the modified tree."""
        ...


class TreeTransformationPipeline:
    """Sequential pipeline for chaining solve tree transformation passes."""

    def __init__(self, passes: Sequence[TreeTransformationPass] | None = None) -> None:
        self._passes: list[TreeTransformationPass] = list(passes or [])

    @property
    def passes(self) -> tuple[TreeTransformationPass, ...]:
        """Sequence of transformation passes in execution order."""
        return tuple(self._passes)

    def add_pass(self, pass_: TreeTransformationPass) -> None:
        """Append a transformation pass to the pipeline."""
        self._passes.append(pass_)

    def apply(self, tree: SolveTree) -> SolveTree:
        """Execute all configured transformation passes sequentially on the tree."""
        current = tree
        for pass_ in self._passes:
            current = pass_.apply(current)
        return current


def _collect_subtree_rooms(node: SolveTreeNode) -> set[int]:
    """Collect all room IDs across a solve tree node and all its descendants."""
    rooms = set(node.rooms)
    for c in node.children:
        rooms.update(_collect_subtree_rooms(c))
    return rooms


def _get_child_poles_or_anchor(child: SolveTreeNode) -> set[int]:
    """Extract boundary poles or anchor room IDs from a child node."""
    if isinstance(child.cut_contract, SplitPairCut):
        return set(child.cut_contract.poles)
    if isinstance(child.cut_contract, ArticulationPointCut):
        return {child.cut_contract.anchor_vnum}
    if isinstance(child.cut_contract, BridgeCut):
        return set(child.cut_contract.bridge_edge)
    if "poles" in child.metadata:
        return set(child.metadata["poles"])
    if "anchor_vnum" in child.metadata:
        return {child.metadata["anchor_vnum"]}
    return set()


def _is_articulation_child(node: SolveTreeNode) -> bool:
    """Return True if node represents an articulation cut child block."""
    if isinstance(node.cut_contract, ArticulationPointCut):
        return True
    if "anchor_vnum" in node.metadata:
        return True
    if node.metadata.get("cut_type") in (CutType.ARTICULATION_POINT, "articulation_point"):
        return True
    return False


def _get_anchor_vnum(node: SolveTreeNode) -> int | None:
    """Extract articulation anchor room ID from node metadata or cut contract."""
    if isinstance(node.cut_contract, ArticulationPointCut):
        return node.cut_contract.anchor_vnum
    if "anchor_vnum" in node.metadata:
        val = node.metadata["anchor_vnum"]
        return int(val) if val is not None else None
    return None


def strip_anchor(node: SolveTreeNode, anchor: int) -> SolveTreeNode:
    """Recursively strip articulation anchor from an SPQR subtree to enforce disjoint ownership."""
    new_rooms = frozenset(node.rooms) - {anchor}
    new_children = tuple(
        strip_anchor(c, anchor) if not _is_articulation_child(c) else c
        for c in node.children
    )
    return SolveTreeNode(
        node_id=node.node_id,
        node_type=node.node_type,
        rooms=new_rooms,
        children=new_children,
        metadata=node.metadata,
    )


def attach_articulation_children(
    spqr_node: SolveTreeNode,
    articulation_children: Sequence[SolveTreeNode],
) -> SolveTreeNode:
    """Attach articulation child blocks to the specific descendant node containing their anchor."""
    if not articulation_children:
        return spqr_node

    if not spqr_node.children:
        return SolveTreeNode(
            node_id=spqr_node.node_id,
            node_type=spqr_node.node_type,
            rooms=spqr_node.rooms,
            children=tuple(list(spqr_node.children) + list(articulation_children)),
            metadata=spqr_node.metadata,
        )

    direct_children: list[SolveTreeNode] = []
    child_to_art_children: dict[str, list[SolveTreeNode]] = collections.defaultdict(list)

    for art_child in articulation_children:
        anchor = _get_anchor_vnum(art_child)
        target_child_id: str | None = None
        if anchor is not None and anchor not in spqr_node.rooms:
            for ch in spqr_node.children:
                if anchor in ch.rooms or anchor in _collect_subtree_rooms(ch):
                    target_child_id = ch.node_id
                    break
        if target_child_id is not None:
            child_to_art_children[target_child_id].append(art_child)
        else:
            direct_children.append(art_child)

    updated_spqr_children: list[SolveTreeNode] = []
    for ch in spqr_node.children:
        if ch.node_id in child_to_art_children:
            updated_ch = attach_articulation_children(ch, child_to_art_children[ch.node_id])
            updated_spqr_children.append(updated_ch)
        else:
            updated_spqr_children.append(ch)

    return SolveTreeNode(
        node_id=spqr_node.node_id,
        node_type=spqr_node.node_type,
        rooms=spqr_node.rooms,
        children=tuple(updated_spqr_children + direct_children),
        metadata=spqr_node.metadata,
    )


class ArticulationChildAttachmentPass:
    """Re-attach articulation child blocks to the specific SPQR node containing their anchor
    and enforce canonical disjoint room ownership by stripping parent anchors."""

    def __init__(self, strip_anchors: bool = True) -> None:
        self.strip_anchors = strip_anchors

    def transform_node(self, node: SolveTreeNode) -> SolveTreeNode:
        """Recursively route articulation children and strip parent anchors on a SolveTreeNode."""
        if node.node_type == TreeNodeType.WORLD_ROOT:
            transformed_children = tuple(self.transform_node(child) for child in node.children)
            return SolveTreeNode(
                node_id=node.node_id if node.node_id else "",
                node_type=node.node_type,
                rooms=node.rooms,
                children=transformed_children,
                metadata=node.metadata,
            )

        # Recursively process children first (bottom-up)
        transformed_children = [self.transform_node(child) for child in node.children]

        # Separate SPQR structural children from articulation children
        spqr_children: list[SolveTreeNode] = []
        art_children: list[SolveTreeNode] = []
        for child in transformed_children:
            if _is_articulation_child(child):
                art_children.append(child)
            else:
                spqr_children.append(child)

        current_node = SolveTreeNode(
            node_id=node.node_id,
            node_type=node.node_type,
            rooms=node.rooms,
            children=tuple(spqr_children),
            metadata=node.metadata,
        )

        if self.strip_anchors:
            parent_anchor = _get_anchor_vnum(node)
            if parent_anchor is not None:
                current_node = strip_anchor(current_node, parent_anchor)

        if art_children:
            current_node = attach_articulation_children(current_node, art_children)

        return current_node

    def apply(self, tree: SolveTree) -> SolveTree:
        """Apply articulation child re-attachment to a SolveTree."""
        if tree.root.node_type == TreeNodeType.EMPTY:
            return tree

        new_root = self.transform_node(tree.root)
        components = new_root.children if new_root.node_type == TreeNodeType.WORLD_ROOT else (new_root,)
        return SolveTree(root=new_root, components=components, graph=tree.graph)


class AnchorStrippingPass:
    """Strip articulation anchors from child subtrees to enforce strict disjoint room ownership."""

    def transform_node(self, node: SolveTreeNode) -> SolveTreeNode:
        """Recursively strip articulation anchors on a SolveTreeNode."""
        if node.node_type == TreeNodeType.WORLD_ROOT:
            transformed_children = tuple(self.transform_node(child) for child in node.children)
            return SolveTreeNode(
                node_id=node.node_id if node.node_id else "",
                node_type=node.node_type,
                rooms=node.rooms,
                children=transformed_children,
                metadata=node.metadata,
            )

        transformed_children = tuple(self.transform_node(child) for child in node.children)
        current = SolveTreeNode(
            node_id=node.node_id,
            node_type=node.node_type,
            rooms=node.rooms,
            children=transformed_children,
            metadata=node.metadata,
        )

        anchor = _get_anchor_vnum(node)
        if anchor is not None:
            current = strip_anchor(current, anchor)

        return current

    def apply(self, tree: SolveTree) -> SolveTree:
        """Apply anchor stripping across all nodes of a SolveTree."""
        if tree.root.node_type == TreeNodeType.EMPTY:
            return tree

        new_root = self.transform_node(tree.root)
        components = new_root.children if new_root.node_type == TreeNodeType.WORLD_ROOT else (new_root,)
        return SolveTree(root=new_root, components=components, graph=tree.graph)


class TransverseSNodeSplitPass:
    """Split series loop (S-node) children with parallel entrance corridors into sibling clusters.

    When a series loop child across poles {p0, p1} connects to parent rooms via parallel entrance
    corridors along the same cardinal axis, an inter-pole span discrepancy along the transverse
    axis forces shear distortion if the child is solved as a single rigid origin component.
    This transformation severs the first transverse internal spine corridor, decomposing the S-node
    into two sibling SolveTreeNode clusters each anchored to its respective parent pole.
    """

    def __init__(self, graph: LayoutGraph | None = None) -> None:
        self._graph = graph

    def transform_node(self, node: SolveTreeNode, graph: LayoutGraph | None = None) -> SolveTreeNode:
        """Recursively split transverse S-nodes on a SolveTreeNode."""
        target_graph = graph or self._graph
        if target_graph is None:
            return node
        return self._split_node(node, target_graph)

    def _split_node(self, node: SolveTreeNode, graph: LayoutGraph) -> SolveTreeNode:
        updated_children: list[SolveTreeNode] = []

        for child in node.children:
            child_processed = self._split_node(child, graph)

            if child_processed.node_type != TreeNodeType.SERIES_LOOP:
                updated_children.append(child_processed)
                continue

            poles: tuple[int, int] | None = None
            if isinstance(child_processed.cut_contract, SplitPairCut):
                poles = child_processed.cut_contract.poles
            elif "poles" in child_processed.metadata:
                raw = child_processed.metadata["poles"]
                if len(raw) == 2:
                    poles = (int(raw[0]), int(raw[1]))

            if poles is None or len(poles) != 2:
                updated_children.append(child_processed)
                continue

            p0, p1 = poles
            if p0 not in node.rooms or p1 not in node.rooms:
                updated_children.append(child_processed)
                continue

            child_all_rooms = _collect_subtree_rooms(child_processed)
            internal_rooms = child_all_rooms - {p0, p1}
            if not internal_rooms:
                updated_children.append(child_processed)
                continue

            p0_dirs: set[Direction] = set()
            p1_dirs: set[Direction] = set()
            for e in graph.edges:
                if e.cut:
                    continue
                if e.src == p0 and e.dst in internal_rooms:
                    p0_dirs.add(Direction(e.direction))
                elif e.dst == p0 and e.src in internal_rooms:
                    p0_dirs.add(Direction(e.direction).invert())

                if e.src == p1 and e.dst in internal_rooms:
                    p1_dirs.add(Direction(e.direction))
                elif e.dst == p1 and e.src in internal_rooms:
                    p1_dirs.add(Direction(e.direction).invert())

            if len(p0_dirs) != 1 or len(p1_dirs) != 1:
                updated_children.append(child_processed)
                continue

            dir_0 = next(iter(p0_dirs))
            dir_1 = next(iter(p1_dirs))
            disp_0 = direction_to_displacement(dir_0)
            disp_1 = direction_to_displacement(dir_1)

            axis_0 = 0 if disp_0[0] != 0 else (1 if disp_0[1] != 0 else (2 if disp_0[2] != 0 else None))
            axis_1 = 0 if disp_1[0] != 0 else (1 if disp_1[1] != 0 else (2 if disp_1[2] != 0 else None))

            if axis_0 is None or axis_1 is None or axis_0 != axis_1:
                updated_children.append(child_processed)
                continue

            entrance_axis = axis_0
            transverse_axis = 1 if entrance_axis == 0 else 0

            spine_verts = set(child_processed.rooms) | {p0, p1}
            adj: dict[int, list[int]] = {v: [] for v in spine_verts}
            for e in graph.edges:
                if not e.cut and e.src in spine_verts and e.dst in spine_verts and {e.src, e.dst} != {p0, p1}:
                    adj[e.src].append(e.dst)
                    adj[e.dst].append(e.src)

            for sub_c in child_processed.children:
                sub_poles = _get_child_poles_or_anchor(sub_c)
                if len(sub_poles) == 2 and sub_poles <= spine_verts:
                    sq0, sq1 = tuple(sub_poles)
                    adj[sq0].append(sq1)
                    adj[sq1].append(sq0)

            queue: list[list[int]] = [[p0]]
            visited: set[int] = {p0}
            spine_path: list[int] | None = None
            while queue:
                curr_path = queue.pop(0)
                u = curr_path[-1]
                if u == p1:
                    spine_path = curr_path
                    break
                for v in adj[u]:
                    if v not in visited:
                        visited.add(v)
                        queue.append(curr_path + [v])

            if not spine_path or len(spine_path) < 3:
                updated_children.append(child_processed)
                continue

            sever_idx: int | None = None
            for i in range(1, len(spine_path) - 2):
                u_i = spine_path[i]
                v_i = spine_path[i + 1]
                matching_edges = [
                    e for e in graph.edges
                    if not e.cut and {e.src, e.dst} == {u_i, v_i}
                ]
                if matching_edges:
                    edge = matching_edges[0]
                    step_dir = Direction(edge.direction) if edge.src == u_i else Direction(edge.direction).invert()
                    disp = direction_to_displacement(step_dir)
                    if disp[transverse_axis] != 0:
                        sever_idx = i
                        break

            if sever_idx is None:
                updated_children.append(child_processed)
                continue

            u_star = spine_path[sever_idx]
            v_star = spine_path[sever_idx + 1]

            adj_severed: dict[int, list[int]] = {v: [] for v in spine_verts}
            for e in graph.edges:
                if (
                    not e.cut
                    and e.src in spine_verts
                    and e.dst in spine_verts
                    and {e.src, e.dst} != {p0, p1}
                    and {e.src, e.dst} != {u_star, v_star}
                ):
                    adj_severed[e.src].append(e.dst)
                    adj_severed[e.dst].append(e.src)
            for sub_c in child_processed.children:
                sub_poles = _get_child_poles_or_anchor(sub_c)
                if len(sub_poles) == 2 and sub_poles <= spine_verts and sub_poles != {u_star, v_star}:
                    sq0, sq1 = tuple(sub_poles)
                    adj_severed[sq0].append(sq1)
                    adj_severed[sq1].append(sq0)

            comp_a: set[int] = set()
            q_a = [p0]
            while q_a:
                curr = q_a.pop()
                if curr not in comp_a:
                    comp_a.add(curr)
                    for nxt in adj_severed[curr]:
                        if nxt not in comp_a:
                            q_a.append(nxt)

            rooms_a = frozenset(comp_a & child_processed.rooms)
            rooms_b = frozenset(child_processed.rooms - rooms_a)

            if not rooms_a or not rooms_b or p0 not in comp_a or p1 in comp_a:
                updated_children.append(child_processed)
                continue

            children_a: list[SolveTreeNode] = []
            children_b: list[SolveTreeNode] = []
            for sub_c in child_processed.children:
                sub_ports = _get_child_poles_or_anchor(sub_c)
                if sub_ports and sub_ports <= rooms_a:
                    children_a.append(sub_c)
                elif sub_ports and sub_ports <= rooms_b:
                    children_b.append(sub_c)
                else:
                    sub_all = _collect_subtree_rooms(sub_c)
                    if len(sub_all & rooms_a) >= len(sub_all & rooms_b):
                        children_a.append(sub_c)
                    else:
                        children_b.append(sub_c)

            meta_a = dict(child_processed.metadata)
            meta_a["cut_type"] = CutType.ARTICULATION_POINT
            meta_a["anchor_vnum"] = p0
            meta_a["cut_contract"] = ArticulationPointCut(anchor_vnum=p0)
            meta_a.pop("poles", None)

            meta_b = dict(child_processed.metadata)
            meta_b["cut_type"] = CutType.ARTICULATION_POINT
            meta_b["anchor_vnum"] = p1
            meta_b["cut_contract"] = ArticulationPointCut(anchor_vnum=p1)
            meta_b.pop("poles", None)

            node_a = SolveTreeNode(
                node_id=f"{child_processed.node_id}_a",
                node_type=TreeNodeType.RIGID_CLUSTER,
                rooms=rooms_a,
                children=tuple(children_a),
                metadata=meta_a,
            )
            node_b = SolveTreeNode(
                node_id=f"{child_processed.node_id}_b",
                node_type=TreeNodeType.RIGID_CLUSTER,
                rooms=rooms_b,
                children=tuple(children_b),
                metadata=meta_b,
            )

            updated_children.extend([node_a, node_b])

        return SolveTreeNode(
            node_id=node.node_id,
            node_type=node.node_type,
            rooms=node.rooms,
            children=tuple(updated_children),
            metadata=node.metadata,
        )

    def apply(self, tree: SolveTree) -> SolveTree:
        """Apply transverse S-node splitting to a SolveTree."""
        graph = self._graph or tree.graph
        if graph is None or tree.root.node_type == TreeNodeType.EMPTY:
            return tree

        new_root = self.transform_node(tree.root, graph)
        components = new_root.children if new_root.node_type == TreeNodeType.WORLD_ROOT else (new_root,)
        return SolveTree(root=new_root, components=components, graph=tree.graph)
