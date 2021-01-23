from anytree import Node, RenderTree

# decision tree of 11 main galaxy classes (super classes)

# initialize nodes
super_class1 = Node('super_class1')
super_class2 = Node('super_class2')
super_class3 = Node('super_class3')
super_class4 = Node('super_class4')
super_class5 = Node('super_class5')
super_class6 = Node('super_class6')
super_class7 = Node('super_class7')
super_class8 = Node('super_class8')
super_class9 = Node('super_class9')
super_class10 = Node('super_class10')
super_class11 = Node('super_class11')

# build tree
super_class1.children = [super_class2, super_class7]
super_class2.children = [super_class3, super_class9]
super_class3.children = [super_class4]
super_class4.children = [super_class5, super_class10]
super_class5.children = [super_class6]
super_class6.children = [super_class8]
super_class7.children = [super_class6]
super_class9.children = []
super_class10.children = [super_class11]
super_class11.children = [super_class5]

print(RenderTree(super_class1))
