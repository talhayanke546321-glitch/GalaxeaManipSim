from types import SimpleNamespace

from galaxea_sim.scripts.eval_pi05_closed_loop import (
    _contact_pair,
    _unexpected_robot_contacts,
)


def _component(name: str):
    return SimpleNamespace(name="", entity=SimpleNamespace(name=name))


def test_contact_pair_resolves_physx_component_owner_names() -> None:
    contact = SimpleNamespace(bodies=[_component("left_arm_link1"), _component("table")])
    assert _contact_pair(contact) == ("left_arm_link1", "table")


def test_collision_metric_excludes_expected_wheel_ground_contact() -> None:
    contacts = [
        SimpleNamespace(bodies=[_component("wheel_motor_link1"), _component("ground")]),
        SimpleNamespace(bodies=[_component("left_arm_link1"), _component("table")]),
    ]
    scene = SimpleNamespace(get_contacts=lambda: contacts)
    assert _unexpected_robot_contacts(scene, {"wheel_motor_link1", "left_arm_link1"}) == {
        ("left_arm_link1", "table")
    }
