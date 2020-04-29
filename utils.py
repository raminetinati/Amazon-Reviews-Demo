


def check_notebook_vars():
    import sys
    # These are the usual ipython objects, including this one you are creating
    ipython_vars = ['In', 'Out', 'exit', 'quit', 'get_ipython', 'ipython_vars']

    # Get a sorted list of the objects and their sizes
    sorted_ = sorted([(x, sys.getsizeof(globals().get(x))) for x in dir() if not x.startswith('_') and x not in sys.modules and x not in ipython_vars], key=lambda x: x[1], reverse=True)
    
    print(sorted_)
    
    
    
### Validate Data
def validate_jsonlines(filename):
    
    with open(filename, 'r') as jfile:
        for line in jfile:
            try:
                line_loaded = json.loads(line)
            except Exception as e:
                print(e)
                print('error in line {}'.format(line))
    
validate_jsonlines('amazon_augmented_train.json')  