import logging
import os

import django
from django.db import transaction
from django.db.models import Count, Q

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "planpincieux.settings")
django.setup()

from ppcx_app.models import Camera, Image  # noqa: E402

logger = logging.getLogger("ppcx")


# -----------------------
# Backend utilities (no interactive I/O)
# -----------------------


def find_duplicate_camera_groups(
    fields: list[str] | None = None,
) -> list[tuple[dict, list[Camera]]]:
    """
    Return list of (field_values_dict, [Camera,...]) for groups where >1 camera share identical values.
    Default fields: camera_name, model, lens, focal_length_mm
    """
    if fields is None:
        fields = ["camera_name", "model", "lens", "focal_length_mm"]

    vals = Camera.objects.values(*fields).annotate(cnt=Count("id")).filter(cnt__gt=1)
    groups: list[tuple[dict, list[Camera]]] = []
    for v in vals:
        q = Q()
        for f in fields:
            val = v.get(f)
            if val is None:
                q &= Q(**{f + "__isnull": True})
            else:
                q &= Q(**{f: val})
        cams = list(Camera.objects.filter(q).order_by("id"))
        if len(cams) > 1:
            groups.append(({f: v.get(f) for f in fields}, cams))
    return groups


def load_cameras_by_ids(ids: list[int]) -> list[Camera]:
    """Return Camera objects for the provided ids in the same order as ids (ignores missing)."""
    if not ids:
        return []
    cams_map = {c.id: c for c in Camera.objects.filter(id__in=ids)}
    return [cams_map[i] for i in ids if i in cams_map]


def merge_cameras_by_ids(
    source_ids: list[int], target_id: int | None = None, commit: bool = True
) -> tuple[int, int]:
    """
    Merge cameras whose ids are in source_ids into target_id.
    If target_id is None, the first id in source_ids is used as target.
    Returns (reassigned_images, deleted_cameras_count).
    """
    if not source_ids:
        return 0, 0
    cams = load_cameras_by_ids(source_ids)
    if not cams:
        return 0, 0
    if target_id is None:
        target = cams[0]
    else:
        target = next((c for c in cams if c.id == target_id), None)
        if target is None:
            raise ValueError("target_id not among provided source_ids")

    dup_ids = [c.id for c in cams if c.id != target.id]
    if not dup_ids:
        return 0, 0

    reassigned = 0
    deleted = 0
    with transaction.atomic():
        reassigned = Image.objects.filter(camera_id__in=dup_ids).update(
            camera_id=target.id
        )
        # delete cameras that now have zero images
        remaining_counts = Image.objects.filter(camera_id__in=dup_ids).values_list(
            "camera_id", flat=True
        )
        remaining_set = set(remaining_counts)
        for cid in dup_ids:
            if cid in remaining_set:
                continue
            Camera.objects.filter(id=cid).delete()
            deleted += 1

        if not commit:
            # force rollback
            raise RuntimeError("dry-run rollback")
    return reassigned, deleted


# -----------------------
# Frontend / CLI helpers (interactive)
# -----------------------


def _print_group_summary(field_values: dict, cams: list[Camera]) -> None:
    svals = ", ".join(f"{k}={repr(v)}" for k, v in field_values.items())
    print(f"\nGroup: {svals}")
    print("Cameras:")
    for c in cams:
        img_count = Image.objects.filter(camera=c).count()
        print(
            f"  id={c.pk} name={c.camera_name!r} model={c.model!r} images={img_count}"
        )


def _prompt_user_choice(cams: list[Camera]) -> int:
    """
    Prompt user to pick which camera id to keep from cams.
    """
    print("Choose target camera to KEEP (other cameras will be merged into it):")
    for c in cams:
        img_count = Image.objects.filter(camera=c).count()
        print(
            f"  [{c.id}] {c.camera_name!r} model={c.model!r} lens={c.lens!r} focal={c.focal_length_mm!r} images={img_count}"
        )
    while True:
        choice = input(
            f"Enter id to keep (or press Enter to keep first [{cams[0].id}]): "
        ).strip()
        if choice == "":
            return cams[0].id
        try:
            cid = int(choice)
            if any(c.id == cid for c in cams):
                return cid
            print("Invalid id; not in the list.")
        except ValueError:
            print("Please enter a numeric id.")


def _parse_id_list(text: str) -> list[int]:
    """
    Parse a comma/space separated list of ints from user input.
    """
    parts = [
        p.strip()
        for p in text.replace("(", "").replace(")", "").split(",")
        if p.strip()
    ]
    ids: list[int] = []
    for p in parts:
        try:
            ids.append(int(p))
        except ValueError:
            continue
    return ids


def interactive_merge(
    *,
    fields: list[str] | None = None,
    dry_run: bool = False,
) -> None:
    """
    Interactive entrypoint:
      - Ask user whether to provide explicit ids (e.g. 23,22) or auto-find duplicates.
      - For explicit ids, prompt for target id (or choose first).
      - For duplicates, iterate groups and prompt per group.
    """
    mode = input(
        "Provide camera ids to merge (comma separated), or press Enter to auto-find duplicates: "
    ).strip()
    if mode:
        ids = _parse_id_list(mode)
        if not ids:
            print("No valid ids provided. Aborting.")
            return
        cams = load_cameras_by_ids(ids)
        if not cams:
            print("No matching cameras found for provided ids.")
            return
        target_id = _prompt_user_choice(cams)
        confirm = (
            input(f"Merge cameras {ids} into {target_id}? [y/N]: ").strip().lower()
        )
        if confirm != "y":
            print("Aborted.")
            return
        try:
            reassigned, deleted = merge_cameras_by_ids(
                ids, target_id=target_id, commit=not dry_run
            )
            print(f"Reassigned {reassigned} images. Deleted {deleted} cameras.")
        except RuntimeError:
            print("Dry-run: no changes committed.")
        return

    # auto-find duplicates
    groups = find_duplicate_camera_groups(fields)
    if not groups:
        print("No duplicate camera groups found.")
        return

    total_reassigned = 0
    total_deleted = 0
    for field_values, cams in groups:
        _print_group_summary(field_values, cams)
        target_id = _prompt_user_choice(cams)
        confirm = (
            input(f"Merge cameras {[c.id for c in cams]} into {target_id}? [y/N]: ")
            .strip()
            .lower()
        )
        if confirm != "y":
            print("Skipping group.")
            continue
        try:
            reassigned, deleted = merge_cameras_by_ids(
                [c.id for c in cams], target_id=target_id, commit=not dry_run
            )
            print(f"Reassigned {reassigned} images. Deleted {deleted} cameras.")
            total_reassigned += reassigned
            total_deleted += deleted
        except RuntimeError:
            print("Dry-run: no changes committed for this group.")
        except Exception as e:
            logger.exception("Failed merging group: %s", e)
            print(f"Error merging group: {e}")

    print(
        f"\nDone. Total reassigned images: {total_reassigned}, total deleted cameras: {total_deleted}"
    )


if __name__ == "__main__":
    # By default run interactive merge; set dry_run=True for a dry-run
    interactive_merge(dry_run=False)
