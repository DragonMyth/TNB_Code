import pydart2 as pydart
import numpy as np


class ReacherTrailWorld(pydart.World):
    def __init__(self, step, skel_path):
        self.trails = []

        super(ReacherTrailWorld, self).__init__(step, skel_path=skel_path)
        # self.trail = []
        self.trail_color = (0, 0, 0)
        self.robo_skel = self.skeletons[-1]

    def step(self, ):
        fingertip = np.array([0.0, -0.25, 0.0])
        self.trails[-1].append([self.robo_skel.bodynodes[2].to_world(fingertip), self.trail_color])
        # self.trail.append(self.skeletons[-1].bodynodes[2].to_world(fingertip))
        super(ReacherTrailWorld, self).step()

    def render_with_ri(self, ri):
        # print("HEHIEHI")

        for trail in self.trails:

            for i in range(len(trail) - 1):
                ri.set_color(trail[0][1][0], trail[0][1][1], trail[0][1][2])

                p0 = trail[i][0]
                p1 = trail[i + 1][0]
                ri.set_line_width(5)
                ri.render_line(p0, p1)

    def reset(self):

        self.trails.append([])
        # self.trail = []
        super(ReacherTrailWorld, self).reset()
