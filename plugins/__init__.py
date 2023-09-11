"""
Load all plugins in folder and serve it in a dict
Each file should have a plugin
1- Get all modules
2 - A return function to return all files

"""
from os.path import dirname, basename, isfile, join
import glob
import inspect
import importlib



# Get all modules in the plugins folder
modules = glob.glob(join(dirname(__file__), "*.py"))
modules= [ '.'+basename(f)[:-3] for f in modules if isfile(f) and not f.endswith('__init__.py')]

# Import the base plugin for comparison purposes
base_plugin=importlib.import_module('.base_plugin',package=".plugins")
Operation=base_plugin.Operation
EntryPoints=base_plugin.EntryPoints
# Add to the __add__ values all the plugins children of base path
plugins_list=[]

for module in modules:
    loaded_module=importlib.import_module(module,package=".plugins")
    clsmembers = inspect.getmembers(loaded_module, inspect.isclass)
    cls=[x for x in clsmembers if (issubclass(x[1],Operation)) and (not x[1]== Operation)]
    plugins_list +=cls

plugins_dict=dict(plugins_list)
# print("Avalaible plugins are :\n","\n".join(list(plugins_dict.keys())))
__all__=[plugins_dict,Operation,EntryPoints]
def return_plugin(name):
    return plugins_dict[name]
