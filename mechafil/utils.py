def validate_qap_method(user_input):
    user_input = user_input.lower()
    if not (user_input == 'basic' or user_input == 'basic-sdm' or user_input == 'tunable'):
        raise ValueError("qap_method must be one of: basic, basic-sdm, tunable")