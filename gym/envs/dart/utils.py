def calc_constraint_force(bodynode1, offset1_dir, bodynode2, offset2_dir, strength=1.0):
    shape1 = bodynode1.shapenodes[0]
    body1_geometry = shape1.shape.size()
    shape2 = bodynode2.shapenodes[0]
    body2_geometry = shape2.shape.size()

    offset1 = offset1_dir * body1_geometry / 2
    offset2 = offset2_dir * body2_geometry / 2

    body1_link_pos_to_world = bodynode1.to_world(offset1)
    body2_link_pos_to_world = bodynode2.to_world(offset2)
    constraint_force_dir = body2_link_pos_to_world - body1_link_pos_to_world
    constraint_force = constraint_force_dir * strength
    return constraint_force, offset1, offset2

def construct_skel_dict(bodynodes):
    node_dict = {}
    for i in range(len(bodynodes)):
        node_dict[bodynodes[i].name] = bodynodes[i]
    return node_dict