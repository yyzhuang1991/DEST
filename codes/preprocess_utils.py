import re 

def transform(twitter_event_str):
    # Replace <*-sub>  in twitter_event with reasonable substitutes
    # Args: 
    #     twitter_event_str: agent,predicate,obj,prep (e.g., "<1stperson-sub>,read,paper,_")
    event = twitter_event_str
    event = " ".join(re.sub("_|,", " ", event).split())
    event = re.sub("<this-sub>", "this", event)
    event = re.sub("<be-sub>", "be", event)
    # 1ST PERSON
    if "<1stperson-sub>" in event: 
        if re.search(" our | we | us ", event): 
            event = re.sub("<1stperson-sub>", "we", event)
        else:
            event = re.sub("<1stperson-sub>", "I", event)

    event = re.sub("<2ndperson-sub>", "you", event)
    
    if "<3rdperson-sub>" in event: 
        if re.search(" his ", event): 
            event = re.sub("<3rdperson-sub>", "he", event)
        elif re.search(" her | hers ", event): 
            event = re.sub("<3rdperson-sub>", "she", event)
        elif re.search(" their | theirs | them ", event):
            event = re.sub("<3rdperson-sub>", "they", event)
        else:
            event = re.sub("<3rdperson-sub>", "he", event)

    # replace @user
    if re.search("@[0-9a-z]+", event): 
        if re.search(" his ", event): 
            event = re.sub("@[0-9a-z]+", "he", event)
        elif re.search(" her | hers ", event): 
            event = re.sub("@[0-9a-z]+", "she", event)
        elif re.search(" their | theirs | them ", event):
            event = re.sub("@[0-9a-z]+", "they", event)
        else:
            event = re.sub("@[0-9a-z]+", "he", event)

    event = re.sub("<neg-sub>", "not", event)
    event = re.sub("<neg>", "", event)

    return event