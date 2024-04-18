import argparse

class Params:
    """
    A container class used to store group of parameters.
    
    Note: Students can ignore this class, this is only used to
    help with grouping of similar arguments at the CLI.

    Attributes
    ----------
    params_keys : List[str]
        a list that holds the all of the name of the parameters
        added to this class

    """
    def __init__(self):
        self.params_keys = []
        
    def __str__(self):
        attr_str = ", ".join([f"{param}={getattr(self,param)}" for param in self.params_keys])
        return f"Params({attr_str})"
    
    def __repr__(self):
        return self.__str__()
    
    def add_parameter(self, parameter_name, parameter_value):
        """
        Adds at runtime a parameter with its respective 
        value to this object.
        
        Parameters
        ----------
        parameter_name : str
            Name of the parameter/variable (identifier)
        parameter_value: object
            Value of the variable
        """
        if len(parameter_name)>1:
            if parameter_name[0] in self.params_keys:
                getattr(self, parameter_name[0]).add_parameter(parameter_name[1:], parameter_value)
            else:
                new_parms = Params()
                new_parms.add_parameter(parameter_name[1:], parameter_value)
                setattr(self, parameter_name[0], new_parms)
                self.params_keys.append(parameter_name[0])

        else:
            setattr(self, parameter_name[0], parameter_value)
            self.params_keys.append(parameter_name[0])
        
    def get_kwargs(self) -> dict:
        """
        Gets all of the parameters stored inside as
        python keyword arguments.
        
        Returns
        ----------
        dict
            python dictionary with variable names as keys
            and their respective value as values.
        """
        kwargs = {}
        for var_name in self.params_keys:
            value = getattr(self, var_name)
            if isinstance(value,Params):
                value = value.get_kwargs()

            kwargs[var_name] = value

        # is a nested dict with only one key? if so, tries to simplify if possible
        if len(kwargs)==1:# and any(isinstance(i,dict) for i in kwargs.values()
            key = list(kwargs.keys())[0]
            if isinstance(kwargs[key], dict):
                return kwargs[key] # just ignores the first dict

        return kwargs

    def get_kwargs_without_defaults(self):
        cli_recorder = CLIRecorder()
        kwargs = self.get_kwargs()
        # remove the arguments that were not specified on the terminal
        return { k:v for k,v in kwargs.items() if k in cli_recorder}


class Singleton(type):
    """
    Python cookbook
    """
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class CLIRecorder(metaclass=Singleton):
    def __init__(self):
        self.args = set()

    def add_arg(self, arg):
        self.args.add(arg.split(".")[-1])

    def __contains__(self, value):
        return value in self.args

class RecordArgument(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        if nargs is not None:
            raise ValueError("nargs not allowed")
        super().__init__(option_strings, dest, **kwargs)
        self.cli_args = CLIRecorder()

    def __call__(self, parser, namespace, values, option_string=None):
        self.cli_args.add_arg(self.dest)
        setattr(namespace, self.dest, values)


def grouping_args(args):
    """
    Auxiliar function to group the arguments group
    the optional arguments that belong to a specific group.
    
    A group is identified with the dot (".") according to the
    following format --<group name>.<variable name> <variable value>.
    
    This method will gather all of the variables that belong to a 
    specific group. Each group is represented as an instance of the
    Params class.
    
    For instance:
        indexer.posting_threshold and indexer.memory_threshold, will be 
        assigned to the same group "indexer", which can be then accessed
        through args.indexer
        
    Parameters
        ----------
        args : argparse.Namespace
            current namespace from argparse
            
    Returns
        ----------
        argparse.Namespace
            modified namespace after performing the grouping
    """
    
    
    namespace_dict = vars(args)
    keys = set(namespace_dict.keys())
    for x in keys:
        if "." in x:
            group_name, *param_name = x.split(".")
            if group_name not in namespace_dict:
                namespace_dict[group_name] = Params()
            #print(namespace_dict.keys())
            namespace_dict[group_name].add_parameter(param_name, namespace_dict[x])
            
            del namespace_dict[x]
    
    return args

def cli_debug_printer(list_parameters, tab=""):
    """
    A helping function to tree print 
    """
    
    params_tree_str = ""

    _tab = " "*(len(tab)-5) + tab[-5:]
    
    for var, value in list_parameters:
        params_tree_str += f"{_tab}{var}: "
        if isinstance(value, Params):
            value_str = cli_debug_printer(sorted(value.get_kwargs().items()), tab+'|--- ')
            params_tree_str += f"\n{_tab}{value_str}"
        elif isinstance(value, dict):
            value_str = cli_debug_printer(sorted(value.items()), tab+'|--- ')
            params_tree_str += f"\n{value_str}"
        else:
            params_tree_str += f"{value}\n"
        #params_tree_str += f"\n"
        
    return params_tree_str