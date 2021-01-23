from graphviz import Digraph 

# Decision tree of 11 main galaxy classes (super classes)

# initialize graph project
galaxy_classes_tree = Digraph(comment='Galaxy Classification Decision Tree')

# initialize nodes 
galaxy_classes_tree.node('class_1', label='Class 1')
galaxy_classes_tree.node('class_2', label='Class 2')
galaxy_classes_tree.node('class_3', label='Class 3')
galaxy_classes_tree.node('class_4', label='Class 4')
galaxy_classes_tree.node('class_5', label='Class 5')
galaxy_classes_tree.node('class_6', label='Class 6')
galaxy_classes_tree.node('class_7', label='Class 7')

# connect nodes (direct graph with no weights)
galaxy_classes_tree.edge('class_1', 'class_7', label='1.1')
galaxy_classes_tree.edge('class_1', 'class_2', label='1.2')

galaxy_classes_tree.edge('class_2', 'class_9', label='2.1')
galaxy_classes_tree.edge('class_2', 'class_3', label='2.2')

galaxy_classes_tree.edge('class_3', 'class_4', label='3.1, 3.2')

galaxy_classes_tree.edge('class_4', 'class_10', label='4.1')
galaxy_classes_tree.edge('class_4', 'class_5', label='4.2')

galaxy_classes_tree.edge('class_5', 'class_6', label='5.1, 5.2, 5.3, 5.4')

galaxy_classes_tree.edge('class_6', 'class_8', label='6.1')

galaxy_classes_tree.edge('class_7', 'class_6', label='7.1, 7.2, 7.3')

galaxy_classes_tree.edge('class_9', 'class_6', label='9.1, 9.2, 9.3')

galaxy_classes_tree.edge('class_10', 'class_11', label='10.1, 10.2, 10.3')

galaxy_classes_tree.edge('class_11', 'class_5', label='11.1, 11.2, 11.3, 11.4, 11.5')

# render tree
galaxy_classes_tree.render(view=True)