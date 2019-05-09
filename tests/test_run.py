"""Check that the run.main() function doesn't throw errors"""
from finn import run


def generate_args(flags_dict):
    """Generate commandline arguments from a dictionary of flags"""
    return [f"--{name}={value}" for name, value in flags_dict.items()]


def default_testing_flags():
    """Flags that make testing easier"""
    return {'epochs': 2, 'use-comet': False}


def add_to_defaults(flags_dict):
    all_flags = default_testing_flags()
    all_flags.update(flags_dict)
    return generate_args(all_flags)


def test_adult_meta_learn():
    run.main(add_to_defaults({
        'dataset': 'adult',
        'meta-learn': False,
        'inv-disc': False,
        'clf-epochs': 3,
    }))
