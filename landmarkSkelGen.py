# create XML
from xml.etree import ElementTree
import xml.dom.minidom

root = ElementTree.Element('root')

child = ElementTree.Element("child")
child.text = "I am a child"

root.append(child)

print(ElementTree.tostring(root))

# parsed_xml = xml.dom.minidom.parseString(ElementTree.tostring(root))
# pretty_xml_as_string = parsed_xml.toprettyxml()
#
# print(pretty_xml_as_string)
num_landmarks = 20
root = ElementTree.Element("SkeletonHolder")
landmarkSize = [0.2, 5, 1]
color = [1, 0, 0, 1]

for i in range(num_landmarks):
    x = i * 5 - int((5 * num_landmarks) / 2)
    trans = [x, 0, -5, 0, 0, 0]
    transStr = ''
    for ele in trans:
        transStr += str(ele) + ' '
    transStr = transStr[:-1]

    sizeStr = ''
    for ele in landmarkSize:
        sizeStr += str(ele) + ' '
    sizeStr = sizeStr[:-1]

    colorStr = ''
    for ele in color:
        colorStr += str(ele) + ' '
    colorStr = colorStr[:-1]

    body = ElementTree.Element('body', name="landmark_" + str(i))
    transformation = ElementTree.Element('transformation')
    transformation.text = transStr
    body.append(transformation)

    visShape = ElementTree.Element('visualization_shape')
    visTrans = ElementTree.Element('transformation')
    visTrans.text = '0 0 0 0 0 0'

    visGeom = ElementTree.Element('geometry')
    box = ElementTree.Element('box')
    visGeomSize = ElementTree.Element('size')
    visGeomSize.text = sizeStr
    box.append(visGeomSize)
    visGeom.append(box)
    colorEle = ElementTree.Element('color')
    colorEle.text = colorStr

    visShape.append(visTrans)
    visShape.append(visGeom)
    visShape.append(colorEle)

    body.append(visShape)
    root.append(body)

for i in range(num_landmarks):
    joint = ElementTree.Element('joint', type="free", name="landmark_joint_" + str(i))
    parent = ElementTree.Element('parent')
    parent.text = 'world'
    joint.append(parent)
    child = ElementTree.Element('child')
    child.text = 'landmark_' + str(i)
    joint.append(child)

    root.append(joint)

parsed_xml = xml.dom.minidom.parseString(ElementTree.tostring(root))
pretty_xml_as_string = parsed_xml.toprettyxml()

print(pretty_xml_as_string)
