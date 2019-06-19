#taken from here: https://gist.github.com/DrDub/6efba6e522302e43d055

import os

import ipywidgets as widgets


class FileBrowser(object):
    def __init__(self):
        self.path = os.getcwd()
        self._update_files()
        
    def _update_files(self):
        self.files = list()
        self.dirs = list()
        if(os.path.isdir(self.path)):
            for f in os.listdir(self.path):
                ff = self.path + "/" + f
                if os.path.isdir(ff):
                    self.dirs.append(f)
                else:
                    self.files.append(f)
        self.files.sort()
        self.dirs.sort()
        
    def widget(self):
        box = widgets.GridBox(layout=widgets.Layout(grid_template_columns='25% 25% 25% 25%'))
                                                   
        
        #box = widgets.VBox()
        self._update(box)
        return box
    
    def _update(self, box):
        
        def on_click(b):
            if b.description == '..':
                self.path = os.path.split(self.path)[0]
            else:
                self.path = self.path + "/" + b.description
                ### fix the problem occurring when go up to root and come back ###
                '''try:
                    if self.path[1] == '/':
                        self.path = self.path[1:];
                except:
                    pass; '''
            self._update_files()
            self._update(box)
        
        buttons = []
        #if self.files:
        button = widgets.Button(description='..', background_color='#d0d0ff')
        button.on_click(on_click)
        buttons.append(button)
        for f in self.dirs:
            button = widgets.Button(description=f, background_color='#d0d0ff')
            button.on_click(on_click)
            buttons.append(button)
        for f in self.files:
            button = widgets.Button(description=f)
            button.on_click(on_click)
            buttons.append(button)

        box.children = tuple(buttons)
        #box.children = tuple([widgets.HTML("<h2>%s</h2>" % (self.path,))] + buttons)

# example usage:
#   f = FileBrowser()
#   f.widget()
#   <interact with widget, select a path>
# in a separate cell:
#   f.path # returns the selected path