import pydart2 as pydart
import numpy as np

BOXNORMALS = {
    'up': [0, 1, 0],
    'down': [0, -1, 0],
    'inward': [0, 0, -1],
    'outward': [0, 0, 1],
    'left': [-1, 0, 0],
    'right': [1, 0, 0]
}


def fastCross(u, v):
    S = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])

    return S.dot(v)


class BaseFluidSimulator(pydart.World):
    def __init__(self, step, skel_path):
        super(BaseFluidSimulator, self).__init__(step, skel_path=skel_path)
        self.forces = np.zeros((len(self.skeletons[-1].bodynodes), 3))
        self.trail = [[], [], [], [], [], []]
        self.debug = False

    def step(self, ):
        # if ((self.t * 2000) % 100 <= 5):
        c = self.skeletons[-1].bodynodes[0].C
        self.trail[0].append(c[0])
        self.trail[1].append(c[1])
        self.trail[2].append(c[2])

        for i in range(len(self.skeletons[-1].bodynodes)):
            # self.forces[i] = self.calcFluidForce(self.skeletons[-1].bodynodes[i])
            self.forces[i] = self.calcFluidForceImproved(self.skeletons[-1].bodynodes[i])

            self.skeletons[-1].bodynodes[i].add_ext_force(self.forces[i])
        super(BaseFluidSimulator, self).step()

    def reset(self):
        self.trail = [[], [], []]
        self.forces = np.zeros((len(self.skeletons[-1].bodynodes), 3))
        super(BaseFluidSimulator, self).reset()

    def calcFluidForceImproved(self, bodynode):
        dq = self.skeletons[-1].dq

        shape = bodynode.shapenodes[0]
        worldCenterPoint = bodynode.to_world([0, 0, 0])

        bodyGeomety = shape.shape.size()

        ang_jac = bodynode.angular_jacobian();
        ang_vel = ang_jac.dot(dq)
        com_vel = bodynode.com_linear_velocity();

        node_linear_velocity_out = com_vel + fastCross(ang_vel, np.array([0, 0, bodyGeomety[2] / 2]))

        world_out_normal = bodynode.to_world(BOXNORMALS['outward']) - worldCenterPoint
        unit_force_out = world_out_normal.dot(node_linear_velocity_out)
        unit_force_out = 0 if unit_force_out < 0 else unit_force_out
        force_out = -bodyGeomety[0] * bodyGeomety[1] * unit_force_out * world_out_normal

        node_linear_velocity_in = com_vel + fastCross(ang_vel, np.array([0, 0, -bodyGeomety[2] / 2]))
        world_in_normal = bodynode.to_world(BOXNORMALS['inward']) - worldCenterPoint
        unit_force_in = world_in_normal.dot(node_linear_velocity_in)
        unit_force_in = 0 if unit_force_in < 0 else unit_force_in
        force_in = -bodyGeomety[0] * bodyGeomety[1] * unit_force_in * world_in_normal

        node_linear_velocity_up = com_vel + fastCross(ang_vel, np.array([0, bodyGeomety[1] / 2, 0]))
        world_up_normal = bodynode.to_world(BOXNORMALS['up']) - worldCenterPoint
        unit_force_up = world_up_normal.dot(node_linear_velocity_up)
        unit_force_up = 0 if unit_force_up < 0 else unit_force_up
        force_up = -bodyGeomety[0] * bodyGeomety[2] * unit_force_up * world_up_normal

        node_linear_velocity_down = com_vel + fastCross(ang_vel, np.array([0, -bodyGeomety[1] / 2, 0]))
        world_down_normal = bodynode.to_world(BOXNORMALS['down']) - worldCenterPoint
        unit_force_down = world_down_normal.dot(node_linear_velocity_down)
        unit_force_down = 0 if unit_force_down < 0 else unit_force_down
        force_down = -bodyGeomety[0] * bodyGeomety[2] * unit_force_down * world_down_normal

        node_linear_velocity_left = com_vel + fastCross(ang_vel, np.array([-bodyGeomety[0] / 2, 0, 0]))
        world_left_normal = bodynode.to_world(BOXNORMALS['left']) - worldCenterPoint
        unit_force_left = world_left_normal.dot(node_linear_velocity_left)
        unit_force_left = 0 if unit_force_left < 0 else unit_force_left
        force_left = -bodyGeomety[1] * bodyGeomety[2] * unit_force_left * world_left_normal

        node_linear_velocity_right = com_vel + fastCross(ang_vel, np.array([bodyGeomety[0] / 2, 0, 0]))
        world_right_normal = bodynode.to_world(BOXNORMALS['right']) - worldCenterPoint
        unit_force_right = world_right_normal.dot(node_linear_velocity_right)
        unit_force_right = 0 if unit_force_right < 0 else unit_force_right
        force_right = -bodyGeomety[1] * bodyGeomety[2] * unit_force_right * world_right_normal

        return force_in + force_out + force_up + force_down + force_left + force_right

    def calcFluidForce(self, bodynode):
        dq = self.skeletons[-1].dq

        shape = bodynode.shapenodes[0]
        worldCenterPoint = bodynode.to_world([0, 0, 0])

        bodyGeomety = shape.shape.size()
        J_out = bodynode.linear_jacobian(offset=[0, 0, bodyGeomety[2] / 2])

        node_linear_velocity_out = J_out.dot(dq)
        world_out_normal = bodynode.to_world(BOXNORMALS['outward']) - worldCenterPoint
        unit_force_out = world_out_normal.dot(node_linear_velocity_out)
        unit_force_out = 0 if unit_force_out < 0 else unit_force_out
        force_out = -bodyGeomety[0] * bodyGeomety[1] * unit_force_out * world_out_normal

        J_in = bodynode.linear_jacobian(offset=[0, 0, -bodyGeomety[2] / 2])
        node_linear_velocity_in = J_in.dot(dq)
        world_in_normal = bodynode.to_world(BOXNORMALS['inward']) - worldCenterPoint
        unit_force_in = world_in_normal.dot(node_linear_velocity_in)
        unit_force_in = 0 if unit_force_in < 0 else unit_force_in
        force_in = -bodyGeomety[0] * bodyGeomety[1] * unit_force_in * world_in_normal

        J_up = bodynode.linear_jacobian(offset=[0, bodyGeomety[1] / 2, 0])
        node_linear_velocity_up = J_up.dot(dq)
        world_up_normal = bodynode.to_world(BOXNORMALS['up']) - worldCenterPoint
        unit_force_up = world_up_normal.dot(node_linear_velocity_up)
        unit_force_up = 0 if unit_force_up < 0 else unit_force_up
        force_up = -bodyGeomety[0] * bodyGeomety[2] * unit_force_up * world_up_normal

        J_down = bodynode.linear_jacobian(offset=[0, -bodyGeomety[1] / 2, 0])
        node_linear_velocity_down = J_down.dot(dq)
        world_down_normal = bodynode.to_world(BOXNORMALS['down']) - worldCenterPoint
        unit_force_down = world_down_normal.dot(node_linear_velocity_down)
        unit_force_down = 0 if unit_force_down < 0 else unit_force_down
        force_down = -bodyGeomety[0] * bodyGeomety[2] * unit_force_down * world_down_normal

        J_left = bodynode.linear_jacobian(offset=[-bodyGeomety[0] / 2, 0, 0])
        node_linear_velocity_left = J_left.dot(dq)
        world_left_normal = bodynode.to_world(BOXNORMALS['left']) - worldCenterPoint
        unit_force_left = world_left_normal.dot(node_linear_velocity_left)
        unit_force_left = 0 if unit_force_left < 0 else unit_force_left
        force_left = -bodyGeomety[1] * bodyGeomety[2] * unit_force_left * world_left_normal

        J_right = bodynode.linear_jacobian(offset=[bodyGeomety[0] / 2, 0, 0])
        node_linear_velocity_right = J_right.dot(dq)
        world_right_normal = bodynode.to_world(BOXNORMALS['right']) - worldCenterPoint
        unit_force_right = world_right_normal.dot(node_linear_velocity_right)
        unit_force_right = 0 if unit_force_right < 0 else unit_force_right
        force_right = -bodyGeomety[1] * bodyGeomety[2] * unit_force_right * world_right_normal

        return force_in + force_out + force_up + force_down + force_left + force_right

    def render_with_ri(self, ri):
        ri.set_color(0.2, 1.0, 0.6)

        po = self.skeletons[0].C
        out = np.array(BOXNORMALS['outward'])
        pd = self.skeletons[0].bodynodes[0].to_world(out * 1.5)
        ri.render_arrow(po, pd, r_base=0.003, head_width=0.003, head_len=0.1)
        trail_len = len(self.trail[0])
        i = trail_len - 1
        while i > 0 and trail_len - i < 1400:
            p0 = np.array([self.trail[0][i - 1], self.trail[1][i - 1], self.trail[2][i - 1]])
            p1 = np.array([self.trail[0][i], self.trail[1][i], self.trail[2][i]])
            ri.render_line(p0, p1)
            i -= 1

        if self.debug:
            for i in range(len(self.skeletons[0].bodynodes)):
                p0 = self.skeletons[0].bodynodes[i].C
                p1 = p0 - 2 * self.forces[i]
                ri.set_color(0.0, 1.0, 0.0)
                ri.render_arrow(p1, p0, r_base=0.03, head_width=0.03, head_len=0.1)


class BaseFluidNoBackSimulator(BaseFluidSimulator):
    def step(self, ):
        c = self.skeletons[-1].bodynodes[0].C
        self.trail[0].append(c[0])
        self.trail[1].append(c[1])
        self.trail[2].append(c[2])

        for i in range(len(self.skeletons[-1].bodynodes)):
            self.forces[i] = self.calcFluidForce(self.skeletons[-1].bodynodes[i])
            self.forces[i, 0] = self.forces[i, 0] * 0.3  # if self.forces[i,0]<=0 else self.forces[i,0]
            self.skeletons[-1].bodynodes[i].add_ext_force(self.forces[i])
        super(BaseFluidSimulator, self).step()


class BaseFluidEnhancedSimulator(BaseFluidSimulator):
    def step(self, ):
        c = self.skeletons[-1].bodynodes[0].C
        self.trail[0].append(c[0])
        self.trail[1].append(c[1])
        self.trail[2].append(c[2])

        for i in range(len(self.skeletons[-1].bodynodes)):
            # self.forces[i] = self.calcFluidForce(self.skeletons[-1].bodynodes[i])
            self.forces[i] = self.calcFluidForceImproved(self.skeletons[-1].bodynodes[i])
            self.forces[i, 0] = self.forces[i, 0] * 1000  # if self.forces[i,0]<=0 else self.forces[i,0]
            self.skeletons[-1].bodynodes[i].add_ext_force(self.forces[i])
        super(BaseFluidSimulator, self).step()


class BaseFluidEnhancedAllDirSimulator(BaseFluidSimulator):
    def step(self, ):
        c = self.skeletons[-1].bodynodes[0].C
        self.trail[0].append(c[0])
        self.trail[1].append(c[1])
        self.trail[2].append(c[2])

        for i in range(1, len(self.skeletons[-1].bodynodes), 1):
            # self.forces[i] = self.calcFluidForce(self.skeletons[-1].bodynodes[i])
            self.forces[i] = self.calcFluidForceImproved(self.skeletons[-1].bodynodes[i])

            self.forces[i] = self.forces[i] * 1000  # if self.forces[i,0]<=0 else self.forces[i,0]
            self.skeletons[-1].bodynodes[i].add_ext_force(self.forces[i])
        super(BaseFluidSimulator, self).step()
