"""Hierarchical solve tree transformation passes and composition pipeline."""

from __future__ import annotations

import collections
from typing import Any, Protocol, Sequence, runtime_checkable

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


def _collect_subtree_severed_edges(node: SolveTreeNode) -> set[tuple[int, int]]:
    """Collect all severed bridge edges declared across a node and all its descendants."""
    edges: set[tuple[int, int]] = set(node.metadata.get("severed_bridge_edges", ()))
    for c in node.children:
        edges.update(_collect_subtree_severed_edges(c))
    return edges


def _get_child_poles_or_anchor(child: SolveTreeNode) -> set[int]:
    """Extract boundary poles or anchor room IDs from a child node."""
    if isinstance(child.cut_contract, SplitPairCut):
        return set(child.cut_contract.poles)
    if isinstance(child.cut_contract, ArticulationPointCut):
        return {child.cut_contract.anchor_vnum}
    if isinstance(child.cut_contract, BridgeCut):
        anc = _get_anchor_vnum(child)
        return {anc} if anc is not None else set(child.cut_contract.bridge_edge)
    if "poles" in child.metadata:
        return set(child.metadata["poles"])
    if "anchor_vnum" in child.metadata:
        return {child.metadata["anchor_vnum"]}
    if "bridge_edge" in child.metadata:
        anc = _get_anchor_vnum(child)
        return {anc} if anc is not None else set(child.metadata["bridge_edge"])
    return set()


def _is_one_cut_child(node: SolveTreeNode) -> bool:
    """Return True if node represents a 1-cut child block (articulation point or bridge)."""
    if isinstance(node.cut_contract, (ArticulationPointCut, BridgeCut)):
        return True
    if "anchor_vnum" in node.metadata or "bridge_edge" in node.metadata:
        return True
    if node.metadata.get("cut_type") in (
        CutType.ARTICULATION_POINT,
        "articulation_point",
        CutType.BRIDGE,
        "bridge",
    ):
        return True
    return False


_is_articulation_child = _is_one_cut_child


def _get_anchor_vnum(node: SolveTreeNode) -> int | None:
    """Extract articulation anchor or bridge parent room ID from node metadata or cut contract."""
    if isinstance(node.cut_contract, ArticulationPointCut):
        return node.cut_contract.anchor_vnum
    if isinstance(node.cut_contract, BridgeCut):
        u, v = node.cut_contract.bridge_edge
        if u not in node.rooms:
            return u
        if v not in node.rooms:
            return v
        return u
    if "anchor_vnum" in node.metadata:
        val = node.metadata["anchor_vnum"]
        return int(val) if val is not None else None
    if "bridge_edge" in node.metadata:
        raw_be = node.metadata["bridge_edge"]
        if len(raw_be) == 2:
            u, v = int(raw_be[0]), int(raw_be[1])
            if u not in node.rooms:
                return u
            if v not in node.rooms:
                return v
            return u
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


def _collect_spqr_branch_rooms(node: SolveTreeNode) -> set[int]:
    """Collect room IDs across an SPQR component and its non-articulation descendants."""
    rooms = set(node.rooms)
    for c in node.children:
        if not _is_articulation_child(c):
            rooms.update(_collect_spqr_branch_rooms(c))
    return rooms



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
                if _is_articulation_child(ch):
                    continue
                if anchor in ch.rooms or anchor in _collect_spqr_branch_rooms(ch):
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



def _is_biconnected_block_root(
    node: SolveTreeNode,
    ancestor_stack: Sequence[SolveTreeNode],
) -> bool:
    """Return True if node is the root of a biconnected block."""
    if node.node_type == TreeNodeType.WORLD_ROOT:
        return False
    if not ancestor_stack:
        return True
    parent = ancestor_stack[-1]
    if parent.node_type == TreeNodeType.WORLD_ROOT:
        return True
    if _is_one_cut_child(node):
        return True
    return False


def _get_block_root(
    node: SolveTreeNode,
    ancestor_stack: Sequence[SolveTreeNode],
) -> SolveTreeNode:
    """Return the nearest ancestor (or node itself) that forms the root of a biconnected block."""
    if _is_biconnected_block_root(node, ancestor_stack):
        return node
    for i in range(len(ancestor_stack) - 1, -1, -1):
        anc = ancestor_stack[i]
        anc_ancestors = ancestor_stack[:i]
        if _is_biconnected_block_root(anc, anc_ancestors):
            return anc
    return node


class TransverseSNodeSplitPass:
    """Split series loop (S-node) children with parallel entrance corridors into sibling clusters.

    When a series loop child across poles {p0, p1} connects to parent rooms via parallel entrance
    corridors along the same cardinal axis, an inter-pole span discrepancy along the transverse
    axis forces shear distortion if the child is solved as a single rigid origin component.
    This transformation severs the first transverse internal spine corridor, decomposing the S-node
    into two sibling SolveTreeNode clusters each anchored to its respective pole.

    If an S-node's immediate container parent does not own its boundary poles (as in split-pair
    containers with empty canonical room sets), the split articulation clusters bubble up to the
    nearest owning ancestor S-node or R-node, or are adopted at the block root level if no higher
    ancestor owns both poles. Any split-pair container left with only a single child is collapsed
    and promoted. S-nodes whose poles are directly connected by an edge preserve tight Euclidean
    cycles and are never severed.
    """

    def __init__(self, graph: LayoutGraph | None = None) -> None:
        self._graph = graph

    def transform_node(
        self,
        node: SolveTreeNode,
        graph: LayoutGraph | None = None,
        ancestor_stack: Sequence[SolveTreeNode] | None = None,
    ) -> SolveTreeNode:
        """Recursively split transverse S-nodes on a SolveTreeNode."""
        target_graph = graph or self._graph
        if target_graph is None:
            return node
        new_node, _ = self._split_node(node, target_graph, list(ancestor_stack or []))
        return new_node

    def _split_node(
        self,
        node: SolveTreeNode,
        graph: LayoutGraph,
        ancestor_stack: list[SolveTreeNode],
    ) -> tuple[SolveTreeNode, list[SolveTreeNode]]:
        original_child_count = len(node.children)
        updated_children: list[SolveTreeNode] = []
        pending_bubbles: list[SolveTreeNode] = []

        new_stack = ancestor_stack + [node]
        for child in node.children:
            ch_processed, ch_bubbles = self._split_node(child, graph, new_stack)
            updated_children.append(ch_processed)
            pending_bubbles.extend(ch_bubbles)

        # Adopt any bubbled children whose articulation anchor is owned by this node
        adopted: list[SolveTreeNode] = []
        still_bubbling: list[SolveTreeNode] = []
        for b in pending_bubbles:
            anchor = _get_anchor_vnum(b)
            if anchor is not None and anchor in node.rooms:
                adopted.append(b)
            else:
                still_bubbling.append(b)

        # If this node is the root of a biconnected block, adopt all remaining
        # bubbled articulation clusters directly to preserve elasticity at the block root.
        is_block_root = _is_biconnected_block_root(node, ancestor_stack)
        if is_block_root and still_bubbling:
            adopted.extend(still_bubbling)
            still_bubbling = []

        updated_children.extend(adopted)

        final_children: list[SolveTreeNode] = []
        split_severed: set[tuple[int, int]] = set()

        for child_processed in updated_children:
            if child_processed.node_type != TreeNodeType.SERIES_LOOP:
                final_children.append(child_processed)
                continue

            poles: tuple[int, int] | None = None
            if isinstance(child_processed.cut_contract, SplitPairCut):
                poles = child_processed.cut_contract.poles
            elif "poles" in child_processed.metadata:
                raw = child_processed.metadata["poles"]
                if len(raw) == 2:
                    poles = (int(raw[0]), int(raw[1]))

            if poles is None or len(poles) != 2:
                final_children.append(child_processed)
                continue

            p0, p1 = poles

            # Identify the owning node: either the immediate parent or closest ancestor owning both poles
            owning_node: SolveTreeNode | None = None
            if p0 in node.rooms and p1 in node.rooms:
                owning_node = node
            else:
                for i in range(len(ancestor_stack) - 1, -1, -1):
                    anc = ancestor_stack[i]
                    if anc.node_type == TreeNodeType.WORLD_ROOT:
                        break
                    if p0 in anc.rooms and p1 in anc.rooms:
                        owning_node = anc
                        break
                    if _is_biconnected_block_root(anc, ancestor_stack[:i]):
                        break

            if owning_node is None:
                if node.node_type == TreeNodeType.WORLD_ROOT:
                    owning_node = node
                else:
                    owning_node = _get_block_root(node, ancestor_stack)

            child_all_rooms = _collect_subtree_rooms(child_processed)
            internal_rooms = child_all_rooms - {p0, p1}
            if not internal_rooms:
                final_children.append(child_processed)
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
                final_children.append(child_processed)
                continue

            dir_0 = next(iter(p0_dirs))
            dir_1 = next(iter(p1_dirs))
            disp_0 = direction_to_displacement(dir_0)
            disp_1 = direction_to_displacement(dir_1)

            axis_0 = 0 if disp_0[0] != 0 else (1 if disp_0[1] != 0 else (2 if disp_0[2] != 0 else None))
            axis_1 = 0 if disp_1[0] != 0 else (1 if disp_1[1] != 0 else (2 if disp_1[2] != 0 else None))

            if axis_0 is None or axis_1 is None or axis_0 != axis_1:
                final_children.append(child_processed)
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
                final_children.append(child_processed)
                continue

            # If boundary poles are directly connected by an edge and the child spine's
            # transverse displacement matches the direct edge displacement, the Euclidean
            # cycle is tight with zero discrepancy; preserve it intact.
            direct_edges = [
                (Direction(e.direction) if e.src == p0 else Direction(e.direction).invert())
                for e in graph.edges
                if not e.cut and {e.src, e.dst} == {p0, p1}
            ]
            if direct_edges:
                direct_disp = direction_to_displacement(direct_edges[0])
                direct_trans = direct_disp[transverse_axis]
                child_trans = 0
                for i in range(len(spine_path) - 1):
                    u_i, v_i = spine_path[i], spine_path[i + 1]
                    s_dirs = [
                        (Direction(e.direction) if e.src == u_i else Direction(e.direction).invert())
                        for e in graph.edges
                        if not e.cut and ((e.src == u_i and e.dst == v_i) or (e.dst == u_i and e.src == v_i))
                    ]
                    if s_dirs:
                        child_trans += direction_to_displacement(s_dirs[0])[transverse_axis]
                if direct_trans == child_trans:
                    final_children.append(child_processed)
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
                else:
                    for sub_c in child_processed.children:
                        sub_poles = _get_child_poles_or_anchor(sub_c)
                        if set(sub_poles) == {u_i, v_i}:
                            sub_rooms = _collect_subtree_rooms(sub_c) - {u_i, v_i}
                            sub_dirs = [
                                (Direction(e.direction) if e.src == u_i else Direction(e.direction).invert())
                                for e in graph.edges
                                if not e.cut and ((e.src == u_i and e.dst in sub_rooms) or (e.dst == u_i and e.src in sub_rooms))
                            ]
                            if sub_dirs:
                                disp = direction_to_displacement(sub_dirs[0])
                                if disp[transverse_axis] != 0:
                                    sever_idx = i
                                    break
                    if sever_idx is not None:
                        break

            if sever_idx is None:
                final_children.append(child_processed)
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
                if len(sub_poles) == 2 and sub_poles <= spine_verts and set(sub_poles) != {u_star, v_star}:
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

            child_rooms = frozenset(child_processed.rooms)
            rooms_a = frozenset(comp_a & child_rooms)
            rooms_b = frozenset(child_rooms - rooms_a)

            if not rooms_a or not rooms_b or p0 not in comp_a or p1 in comp_a:
                final_children.append(child_processed)
                continue

            children_a: list[SolveTreeNode] = []
            children_b: list[SolveTreeNode] = []
            for sub_c in child_processed.children:
                sub_anc = _get_anchor_vnum(sub_c)
                if sub_anc is not None:
                    if sub_anc in rooms_a:
                        children_a.append(sub_c)
                    elif sub_anc in rooms_b:
                        children_b.append(sub_c)
                    else:
                        sub_all = _collect_subtree_rooms(sub_c)
                        if len(sub_all & rooms_a) >= len(sub_all & rooms_b):
                            children_a.append(sub_c)
                        else:
                            children_b.append(sub_c)
                else:
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

            # Under Disjoint Room Ownership, p0 and p1 are owned by the ancestor node.
            rooms_a = frozenset(rooms_a - {p0, p1})
            rooms_b = frozenset(rooms_b - {p0, p1})

            subtree_rooms_a = set(rooms_a)
            for c in children_a:
                subtree_rooms_a.update(_collect_subtree_rooms(c))
            subtree_rooms_a.discard(p0)
            subtree_rooms_a.discard(p1)

            subtree_rooms_b = set(rooms_b)
            for c in children_b:
                subtree_rooms_b.update(_collect_subtree_rooms(c))
            subtree_rooms_b.discard(p0)
            subtree_rooms_b.discard(p1)

            # Identify unique entrance graph edge between p0 and subtree_rooms_a
            edges_p0 = [
                e
                for e in graph.edges
                if not e.cut
                and (
                    (e.src == p0 and e.dst in subtree_rooms_a)
                    or (e.dst == p0 and e.src in subtree_rooms_a)
                )
            ]
            u_entries = {e.dst if e.src == p0 else e.src for e in edges_p0}
            meta_a = dict(child_processed.metadata)
            u_entry_external: int | None = None
            if len(u_entries) == 1:
                u_entry = next(iter(u_entries))
                edge_p0 = next((e for e in edges_p0 if e.src == p0 and e.dst == u_entry), None)
                if edge_p0 is not None:
                    edge_dir_a = int(edge_p0.direction)
                    edge_weight_a = edge_p0.weight
                else:
                    edge_rev = next(e for e in edges_p0 if e.src == u_entry and e.dst == p0)
                    edge_dir_a = int(Direction(edge_rev.direction).invert())
                    edge_weight_a = edge_rev.weight

                meta_a["cut_type"] = CutType.BRIDGE
                meta_a["cut_contract"] = BridgeCut(
                    bridge_edge=(p0, u_entry),
                    direction=edge_dir_a,
                    weight=edge_weight_a,
                )
                meta_a["bridge_edge"] = (p0, u_entry)
                meta_a["direction"] = edge_dir_a
                meta_a["weight"] = edge_weight_a
                meta_a.pop("anchor_vnum", None)
                meta_a.pop("poles", None)
                if u_entry not in rooms_a:
                    u_entry_external = u_entry
            else:
                meta_a["cut_type"] = CutType.ARTICULATION_POINT
                meta_a["anchor_vnum"] = p0
                meta_a["cut_contract"] = ArticulationPointCut(anchor_vnum=p0)
                meta_a.pop("poles", None)

            # Identify unique entrance graph edge between p1 and subtree_rooms_b
            edges_p1 = [
                e
                for e in graph.edges
                if not e.cut
                and (
                    (e.src == p1 and e.dst in subtree_rooms_b)
                    or (e.dst == p1 and e.src in subtree_rooms_b)
                )
            ]
            v_entries = {e.dst if e.src == p1 else e.src for e in edges_p1}
            meta_b = dict(child_processed.metadata)
            v_entry_external: int | None = None
            if len(v_entries) == 1:
                v_entry = next(iter(v_entries))
                edge_p1 = next((e for e in edges_p1 if e.src == p1 and e.dst == v_entry), None)
                if edge_p1 is not None:
                    edge_dir_b = int(edge_p1.direction)
                    edge_weight_b = edge_p1.weight
                else:
                    edge_rev = next(e for e in edges_p1 if e.src == v_entry and e.dst == p1)
                    edge_dir_b = int(Direction(edge_rev.direction).invert())
                    edge_weight_b = edge_rev.weight

                meta_b["cut_type"] = CutType.BRIDGE
                meta_b["cut_contract"] = BridgeCut(
                    bridge_edge=(p1, v_entry),
                    direction=edge_dir_b,
                    weight=edge_weight_b,
                )
                meta_b["bridge_edge"] = (p1, v_entry)
                meta_b["direction"] = edge_dir_b
                meta_b["weight"] = edge_weight_b
                meta_b.pop("anchor_vnum", None)
                meta_b.pop("poles", None)
                if v_entry not in rooms_b:
                    v_entry_external = v_entry
            else:
                meta_b["cut_type"] = CutType.ARTICULATION_POINT
                meta_b["anchor_vnum"] = p1
                meta_b["cut_contract"] = ArticulationPointCut(anchor_vnum=p1)
                meta_b.pop("poles", None)

            descendant_severed = _collect_subtree_severed_edges(child_processed)

            severed_ports_a: set[int] = set()
            severed_ports_b: set[int] = set()
            cross_edges: list[tuple[int, int]] = []

            for e in graph.edges:
                if e.cut:
                    continue
                edge_pair = (min(e.src, e.dst), max(e.src, e.dst))
                if edge_pair in descendant_severed:
                    continue
                if e.src in subtree_rooms_a and e.dst in subtree_rooms_b:
                    severed_ports_a.add(e.src)
                    severed_ports_b.add(e.dst)
                    cross_edges.append(edge_pair)
                elif e.src in subtree_rooms_b and e.dst in subtree_rooms_a:
                    severed_ports_b.add(e.src)
                    severed_ports_a.add(e.dst)
                    cross_edges.append(edge_pair)

            if u_star in subtree_rooms_a:
                severed_ports_a.add(u_star)
            elif u_star in subtree_rooms_b:
                severed_ports_b.add(u_star)

            if v_star in subtree_rooms_a:
                severed_ports_a.add(v_star)
            elif v_star in subtree_rooms_b:
                severed_ports_b.add(v_star)

            if not cross_edges and {u_star, v_star}:
                cross_edges.append((min(u_star, v_star), max(u_star, v_star)))

            severed_edges = tuple(sorted(set(cross_edges)))

            existing_ports = set(child_processed.metadata.get("boundary_ports", ()))
            severed_ports_a.update(existing_ports & subtree_rooms_a)
            severed_ports_b.update(existing_ports & subtree_rooms_b)
            for c in children_a:
                if "boundary_ports" in c.metadata:
                    severed_ports_a.update(c.metadata["boundary_ports"])
            for c in children_b:
                if "boundary_ports" in c.metadata:
                    severed_ports_b.update(c.metadata["boundary_ports"])
            if u_entry_external is not None:
                severed_ports_a.add(u_entry_external)
            if v_entry_external is not None:
                severed_ports_b.add(v_entry_external)

            ports_a = tuple(sorted(severed_ports_a - rooms_a))
            ports_b = tuple(sorted(severed_ports_b - rooms_b))
            if ports_a:
                meta_a["boundary_ports"] = ports_a
            if ports_b:
                meta_b["boundary_ports"] = ports_b

            severed_edges_a = set(severed_edges)
            for c in children_a:
                if "severed_bridge_edges" in c.metadata:
                    severed_edges_a.update(c.metadata["severed_bridge_edges"])
            severed_edges_b = set(severed_edges)
            for c in children_b:
                if "severed_bridge_edges" in c.metadata:
                    severed_edges_b.update(c.metadata["severed_bridge_edges"])

            if severed_edges_a:
                meta_a["severed_bridge_edges"] = tuple(sorted(severed_edges_a))
            if severed_edges_b:
                meta_b["severed_bridge_edges"] = tuple(sorted(severed_edges_b))

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

            if owning_node is node:
                final_children.extend([node_a, node_b])
                if severed_edges_a:
                    split_severed.update(severed_edges_a)
                if severed_edges_b:
                    split_severed.update(severed_edges_b)
            else:
                still_bubbling.extend([node_a, node_b])

        # If container was left with 1 child after a split, collapse and promote it
        if (
            (node.node_type == TreeNodeType.SPLIT_PAIR_CONTAINER or not node.rooms)
            and node.node_type != TreeNodeType.WORLD_ROOT
            and original_child_count > 1
            and len(final_children) == 1
            and not adopted
        ):
            return final_children[0], still_bubbling

        if node.node_type == TreeNodeType.WORLD_ROOT and still_bubbling:
            final_children.extend(still_bubbling)
            still_bubbling = []

        new_metadata = dict(node.metadata)
        node_severed = set(new_metadata.get("severed_bridge_edges", ()))
        node_severed.update(split_severed)
        for b in adopted:
            if "severed_bridge_edges" in b.metadata:
                node_severed.update(b.metadata["severed_bridge_edges"])
        if node_severed:
            new_metadata["severed_bridge_edges"] = tuple(sorted(node_severed))

        return (
            SolveTreeNode(
                node_id=node.node_id,
                node_type=node.node_type,
                rooms=node.rooms,
                children=tuple(final_children),
                metadata=new_metadata,
            ),
            still_bubbling,
        )

    def apply(self, tree: SolveTree) -> SolveTree:
        """Apply transverse S-node splitting to a SolveTree."""
        graph = self._graph or tree.graph
        if graph is None or tree.root.node_type == TreeNodeType.EMPTY:
            return tree

        new_root, _ = self._split_node(tree.root, graph, [])
        components = new_root.children if new_root.node_type == TreeNodeType.WORLD_ROOT else (new_root,)
        return SolveTree(root=new_root, components=components, graph=tree.graph)


def __getattr__(name: str) -> Any:
    if name == "BlockCutDecompositionPass":
        from romutil.layout.hierarchy.block_cut import BlockCutDecompositionPass

        return BlockCutDecompositionPass
    if name == "SpqrDecompositionPass":
        from romutil.layout.hierarchy.spqr import SpqrDecompositionPass

        return SpqrDecompositionPass
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
