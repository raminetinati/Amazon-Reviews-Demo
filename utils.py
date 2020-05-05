#Author: Ramine Tinati


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



def count_s3_obj_lines(configs, global_vars, entry):
        
    s3 = boto3.client('s3')

    resp = s3.select_object_content(
        Bucket=configs['bucket_name'],
        Key=entry['path_with_prefix'],
        ExpressionType='SQL',
        Expression="SELECT count(*) FROM s3object s",
        InputSerialization = {'CSV':
                              {"FileHeaderInfo": "Use", 
                               "AllowQuotedRecordDelimiter": True,
                               "QuoteEscapeCharacter":"\\",
                              }, 
                              'CompressionType': 'NONE'},
        OutputSerialization = {'CSV':{}},
    )
    
    for event in resp['Payload']:
        if 'Records' in event:
            records = event['Records']['Payload'].decode('utf-8')
#             print('Rows:',records)
            return(int(records))


def prep_data_for_supervised_blazing_text_csv(df, train_file_output_name, test_file_output_name):
    
    
    label_prefix = "__label__"    
    labels = (label_prefix + df['product_category']).tolist()
    #and tokenized words
    tmp = df['review_body_processed']
    xs = []
    for entry in tmp:
        res = str(entry).strip('][').split(', ') 
        res = ' '.join(res)
        xs.append(res)
    
   
    #split the data into test and train for supervised mode
    X_train, X_test, y_train, y_test = train_test_split(
        xs, labels, random_state = 0)
    
    train_prepped = []
    #train
    for i in range(0, len(X_train)):
        
        row = str(y_train[i]) + " " + str(X_train[i])
        train_prepped.append([row])
#     print('Train Processed Data: {}'.format(train_prepped[0]))
    
    test_prepped = []
    #train
    for i in range(0, len(X_test)):
        row = str(y_test[i]) + " " + str(X_test[i])
        test_prepped.append([row])
    print('')
#     print('Test Processed Data: {}'.format(test_prepped[0]))
    
    with open(train_file_output_name, 'w') as csvoutfile:
        csv_writer = csv.writer(csvoutfile, delimiter=' ', 
                                lineterminator='\n',  
                                escapechar=' ', 
                                quoting=csv.QUOTE_NONE)
        csv_writer.writerows(train_prepped)

    with open(test_file_output_name, 'w') as csvoutfile:
        csv_writer = csv.writer(csvoutfile, delimiter=' ', 
                                lineterminator='\n',  
                                escapechar=' ', 
                                quoting=csv.QUOTE_NONE)        
        csv_writer.writerows(test_prepped)
        
    return True
