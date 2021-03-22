from nifti_processor import NIFTIProcessor

if __name__ == '__main__':
    task = NIFTIProcessor(cli=True)
    task.run()