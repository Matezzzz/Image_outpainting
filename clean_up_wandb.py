import argparse

import wandb


parser = argparse.ArgumentParser()


parser.add_argument("--artifact_path", default="matezzzz/image_outpainting_maskgit/run_xk236zct_model", type=str, help="The artifact to delete")
parser.add_argument("--delete_from", default=0, type=int, help="The version from which to start deleting")
parser.add_argument("--delete_to", default=0, type=int, help="The last version to delete")
parser.add_argument("--keep_each", default=20, type=int, help="Keep each version where (ver % keep_each) == 0")




def clean_up_wandb(artifact_path, delete_from, delete_to, keep_each):
    """Delete unnecessary artifacts (old models/code) from wandb"""
    api = wandb.Api()

    #go over all versions to delete
    for i in range(delete_from, delete_to+1):
        #if I want to keep this one, keep it
        if i % keep_each == 0:
            print ("Keeping", i)
            continue
        try:
            #try getting the given version of the artifact
            artifact = api.artifact(f'{artifact_path}:v{i}')
        except wandb.errors.CommError:
            #if we failed to find it, print an error
            print ("Can't find", i)
            continue
        #delete the artifact if it was found
        print ("Deleting", i)
        artifact.delete(delete_aliases=True)
        artifact.delete()


if __name__ == "__main__":
    _args = parser.parse_args([])
    clean_up_wandb(_args.artifact_path, _args.delete_from, _args.delete_to, _args.keep_each)
