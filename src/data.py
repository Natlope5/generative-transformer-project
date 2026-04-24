def load_samsum_subset(split="test", num_examples=5, seed=42):
    """
    Return a tiny, built-in 'fake SAMSum' dataset so we don't rely on Hugging Face datasets/pyarrow.
    """
    dialogues = [
        "Sam: Hey, are we still on for the meeting at 3?\nAlex: Yes, but can we move it to 3:30?\nSam: That works for me.\nAlex: Great, see you then.",
        "Mom: Did you finish your homework?\nKid: Almost, I have one assignment left.\nMom: Please finish before dinner.\nKid: Okay, I will.",
        "Alice: The deployment failed again.\nBob: I saw, the database migration script crashed.\nAlice: Can you roll it back and open a ticket?\nBob: Already rolling back and logging an incident.",
        "Manager: The client call is rescheduled to tomorrow.\nEngineer: Morning or afternoon?\nManager: 10 AM sharp.\nEngineer: Got it, I'll update my calendar.",
        "Friend1: Want to watch a movie tonight?\nFriend2: Sure, let's pick something on Netflix.\nFriend1: I'll send you a few options.\nFriend2: Sounds good."
    ]

    references = [
        "Sam and Alex reschedule their meeting from 3:00 to 3:30.",
        "A mother reminds her child to finish homework before dinner.",
        "Alice and Bob handle a failed deployment by rolling back and opening an incident.",
        "A manager moves a client call to 10 AM the next day and the engineer updates their calendar.",
        "Two friends plan to watch a movie together and choose something on Netflix."
    ]

    dialogues = dialogues[:num_examples]
    references = references[:num_examples]
    return dialogues, references