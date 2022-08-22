header = """<?xml version="1.0"?>\n"""
tail = """</robot>"""


class Link:
    """It is used to generate link code"""

    def __init__(self, name):
        self.link_body = []
        self.blank = "  "
        name_body = "<link name=" + "\"" + name + "\">\n"
        self.link_body.append(self.blank)
        self.link_body.append(name_body)

    def inertial(self, origin: list, inertia: list, mass):
        """

        :param origin: tuple, shape=(6,)
        :param inertia: tuple, shape=(6,).0~3:xyz,3~6:rpy.
        :param mass: float
        :return:
        """
        self.link_body.append(self.blank + self.blank + "<inertial>\n")
        self.link_body.append(self.blank + self.blank + self.blank)
        origin_ = "<origin xyz=\"{} {} {}\" rpy=\"{} {} {}\"/>\n".format(origin[0], origin[1], origin[2], origin[3],
                                                                         origin[4], origin[5])
        self.link_body.append(origin_)
        self.link_body.append(self.blank + self.blank + self.blank)
        inertia_ = "<inertia ixx=\"{}\" ixy=\"{}\" ixz=\"{}\" iyy=\"{}\" iyz=\"{}\" izz=\"{}\"/>\n".format(inertia[0],
                                                                                                           inertia[1],
                                                                                                           inertia[2],
                                                                                                           inertia[3],
                                                                                                           inertia[4],
                                                                                                           inertia[5])
        self.link_body.append(inertia_)
        self.link_body.append(self.blank + self.blank + self.blank)
        mass_ = "<mass value=\"{}\"/>\n".format(mass)
        self.link_body.append(mass_)
        self.link_body.append(self.blank + self.blank)
        self.link_body.append("</inertial>\n")

    def visual(self, origin, color, size, file=None):
        self.link_body.append(self.blank + self.blank)
        name_ = "<visual>\n"
        self.link_body.append(name_)
        self.link_body.append(self.blank + self.blank + self.blank)
        origin_ = "<origin xyz=\"{} {} {}\" rpy=\"{} {} {}\"/>\n".format(origin[0], origin[1], origin[2], origin[3],
                                                                         origin[4], origin[5])
        self.link_body.append(origin_)
        self.link_body.append(self.blank + self.blank + self.blank)
        self.link_body.append("<geometry>\n")
        self.link_body.append(self.blank + self.blank + self.blank + self.blank)
        if file is None:
            if len(size) == 3:
                mesh = "<box size=\"{} {} {}\"/>\n".format(size[0], size[1], size[2])
            else:
                mesh = "<cylinder radius=\"{}\" length=\"{}\"/>\n".format(size[0], size[1])
        else:
            mesh = "<mesh filename=\"{}\" scale=\"{} {} {}\"/>\n".format(file, size[0], size[1], size[2])

        self.link_body.append(mesh)
        self.link_body.append(self.blank + self.blank + self.blank)
        self.link_body.append("</geometry>\n")

        self.link_body.append(self.blank + self.blank + self.blank)
        color_ = "<material name=\"color\">\n"
        self.link_body.append(color_)
        self.link_body.append(self.blank + self.blank + self.blank + self.blank)
        self.link_body.append("<color rgba=\"{} {} {} {}\"/>\n".format(color[0], color[1], color[2], color[3]))
        self.link_body.append(self.blank + self.blank + self.blank)
        self.link_body.append("</material>\n")
        self.link_body.append(self.blank + self.blank)
        self.link_body.append("</visual>\n")

    def collision(self, origin, size):
        self.link_body.append(self.blank + self.blank)
        self.link_body.append("<collision>\n")
        self.link_body.append(self.blank + self.blank + self.blank)
        origin_ = "<origin xyz=\"{} {} {}\" rpy=\"{} {} {}\"/>\n".format(origin[0], origin[1], origin[2], origin[3],
                                                                         origin[4], origin[5])
        self.link_body.append(origin_)
        self.link_body.append(self.blank + self.blank + self.blank)
        self.link_body.append("<geometry>\n")
        self.link_body.append(self.blank + self.blank + self.blank + self.blank)
        if len(size) == 3:
            mesh = "<box size=\"{} {} {}\"/>\n".format(size[0], size[1], size[2])
        else:
            mesh = "<cylinder radius=\"{}\" length=\"{}\"/>\n".format(size[0], size[1])
        self.link_body.append(mesh)
        self.link_body.append(self.blank + self.blank + self.blank)
        self.link_body.append("</geometry>\n")
        self.link_body.append(self.blank + self.blank)
        self.link_body.append("</collision>\n")

    def contact(self,value):
        self.link_body.append(self.blank + self.blank)
        self.link_body.append("<contact>\n")
        self.link_body.append(self.blank + self.blank+self.blank)
        self.link_body.append("<lateral_fraction value=\"{}\"/>\n".format(value))
        self.link_body.append(self.blank + self.blank)
        self.link_body.append("</contact>\n")

    def get_block(self):
        self.link_body.append(self.blank)
        self.link_body.append("</link>\n")
        urdf_code = ""
        for urdf in self.link_body:
            urdf_code += urdf
        return urdf_code


class URDF:
    """
    A tool helps you manage urdf files.
    """

    def __init__(self, name):
        self.urdf_body = []
        self.urdf_body.append(header)
        # name_ = "<robot name=" + """\"""" + name + """\"""" + ">\n"
        name_ = "<robot name=\"{}\">\n".format(name)
        self.urdf_body.append(name_)

    def save(self, file_name):
        self.urdf_body.append(tail)
        urdf_code = ""
        for urdf in self.urdf_body:
            urdf_code += urdf
            # urdf_code += "\n"
        with open(file_name, "w") as f:
            f.write(urdf_code)

    def add_link(self, link: Link):
        self.urdf_body.append(link.get_block())


if __name__ == "__main__":
    import numpy as np

    landing_urdf = URDF("landing area")
    landing_link = Link('baseLink')
    origin = np.zeros(6)
    origin_0 = origin.tolist()
    origin[5] = np.random.uniform(-1.5, 1.5, 1)[0]
    origin = origin.tolist()
    landing_link.inertial(origin_0, origin_0, 0)
    color = np.random.uniform(0, 1, 4)
    color[3] = 1
    color = color.tolist()
    size = np.random.uniform(0.4, 0.6, 3)
    size[2] = 0.01
    size = size.tolist()
    landing_link.visual(origin, color, size)
    landing_urdf.add_link(landing_link)
    landing_urdf.save('landing.urdf')
