"""E2E Attack to create adversarial documents."""

from PIL import Image
from attacks.pix2struct_attack import e2e_attack_pix2struct
from models.pix2struct import Pix2StructModel, Pix2StructModelProcessor
from models.processing.pix2struct_processor import Pix2StructImageProcessor

def single_question_attack():
    image = Image.open("data/inputs/0a0a0792728288619a600f55_0.jpg")
    question = "How much is the total?"
    target = "$0.00"
    
    processor = Pix2StructImageProcessor()
    autoprocessor = Pix2StructModelProcessor()
    model = Pix2StructModel()

    rec = e2e_attack_pix2struct(
        model,
        processor,
        autoprocessor,
        image,
        question,
        target,
        config_file="config.ini",
    )
    
    rec.save("data/results/adv_e2e/single_optimized.jpg")

def multiple_questions_attack():
    # Prepare input (image, question+target pair)
    image = Image.open("data/inputs/0a0a0792728288619a600f55_0.jpg")
    questions = ["How much is the total?", "What is the invoice number?"]
    targets = ["$0.00","0"]
    
    processor = Pix2StructImageProcessor()
    autoprocessor = Pix2StructModelProcessor()
    model = Pix2StructModel()

    rec = e2e_attack_pix2struct(
        model,
        processor,
        autoprocessor,
        image,
        questions,
        targets,
        config_file="config.ini",
    )
    
    rec.save("data/results/adv_e2e/multiple_optimized.jpg")

def denial_of_service_attack():
    image = Image.open("data/inputs/0a0a0792728288619a600f55_0.jpg")
    questions = ["How much is the total?", 
                 "What is the invoice number?",
                 "Are there comments?"]
    ground_truth_target = ["$2,150.00",
                           "8257383",
                           "BRAD PHILIPPS"]
    
    processor = Pix2StructImageProcessor()
    autoprocessor = Pix2StructModelProcessor()
    model = Pix2StructModel()

    rec = e2e_attack_pix2struct(
        model,
        processor,
        autoprocessor,
        image,
        questions,
        ground_truth_target,
        config_file="config.ini",
        is_targeted=False
    )
    
    rec.save("data/results/adv_e2e/dos_optimized.jpg")

if __name__ == '__main__':
    print('Running single question attack')
    single_question_attack()
    
    print('Running DoS attack')
    denial_of_service_attack()

    print('Running multiple questions attack')
    multiple_questions_attack()
