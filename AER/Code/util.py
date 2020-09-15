###
# Utility function for terminal execution of files
###

def parse_arguments(args):
    arg_list=[]
    kwargs = dict()
    for arg in args:
        if arg.startswith('--'):
            key, val = arg[2:].split('=')
            if val[0] == '[': val = val[1:-1].split(',')
            kwargs[key] = val
        elif arg.startswith('-'):
            arg_list.append(arg[1:])
    return args, kwargs