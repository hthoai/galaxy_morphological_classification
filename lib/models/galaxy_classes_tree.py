from graphviz import Digraph 

# initialize graph project
galaxy_classes_tree = Digraph(comment='Galaxy Classification Decision Tree')

# initialize nodes 
galaxy_classes_tree.node('Class_1', label='Class 1')
galaxy_classes_tree.node('Class_2', label='Class 2')
galaxy_classes_tree.node('Class_3', label='Class 3')
galaxy_classes_tree.node('Class_4', label='Class 4')
galaxy_classes_tree.node('Class_5', label='Class 5')
galaxy_classes_tree.node('Class_6', label='Class 6')
galaxy_classes_tree.node('Class_7', label='Class 7')

# connect nodes with labels
galaxy_classes_tree.edge('Class_1', 'Class_7', label='1.1')
galaxy_classes_tree.edge('Class_1', 'Class_2', label='1.2')

galaxy_classes_tree.edge('Class_2', 'Class_9', label='2.1')
galaxy_classes_tree.edge('Class_2', 'Class_3', label='2.2')

galaxy_classes_tree.edge('Class_3', 'Class_4', label='3.1, 3.2')

galaxy_classes_tree.edge('Class_4', 'Class_10', label='4.1')
galaxy_classes_tree.edge('Class_4', 'Class_5', label='4.2')

galaxy_classes_tree.edge('Class_5', 'Class_6', label='5.1, 5.2, 5.3, 5.4')

galaxy_classes_tree.edge('Class_6', 'Class_8', label='6.1')

galaxy_classes_tree.edge('Class_7', 'Class_6', label='7.1, 7.2, 7.3')

galaxy_classes_tree.edge('Class_9', 'Class_6', label='9.1, 9.2, 9.3')

galaxy_classes_tree.edge('Class_10', 'Class_11', label='10.1, 10.2, 10.3')

galaxy_classes_tree.edge('Class_11', 'Class_5', label='11.1, 11.2, 11.3, 11.4, 11.5')

# render tree
galaxy_classes_tree.render(view=True)