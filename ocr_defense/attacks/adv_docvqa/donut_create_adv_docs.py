"""Attack to create adversarial documents."""

from PIL import Image

from attacks.donut_attack import attack_donut
from models.donut import DonutModel, DonutModelProcessor
from models.processing.donut_processor import DonutImageProcessor
from transformers import AutoProcessor

def single_question_attack():
    image = Image.open("data/inputs/0a0a0792728288619a600f55_0.jpg")
    question = "How much is the total?"
    target = "$0.00"
    
    processor : DonutImageProcessor = DonutImageProcessor()
    auto_processor = DonutModelProcessor()
    model : DonutModel = DonutModel()

    rec = attack_donut(
        model,
        processor,
        auto_processor,
        image,
        question,
        target,
        config_file="config.ini",
        is_targeted=True
    )
    rec.save("data/results/adv_donut/single_optimized.jpg")

def multiple_questions_attack():
    image = Image.open("data/inputs/0a0a0792728288619a600f55_0.jpg")
    questions = ["How much is the total?", "What is the invoice number?"]
    targets = ["$0.00","0"]
    
    processor : DonutImageProcessor = DonutImageProcessor()
    auto_processor = DonutModelProcessor()
    model : DonutModel = DonutModel()

    rec = attack_donut(
        model,
        processor,
        auto_processor,
        image,
        questions,
        targets,
        config_file="config.ini",
        is_targeted=True
    )
    
    rec.save("data/results/adv_donut/multiple_optimized.jpg")

def denial_of_answer_attack():
    image = Image.open("data/inputs/0a0a0792728288619a600f55_0.jpg")
    questions = ["How much is the total?", 
                 "What is the invoice number?",
                 "Are there comments?"]
    ground_truth_target = ["$1,827.50",
                           "8257383",
                           "brad philips"]
    
    processor : DonutImageProcessor = DonutImageProcessor()
    auto_processor = DonutModelProcessor()
    model : DonutModel = DonutModel()

    rec = attack_donut(
        model,
        processor,
        auto_processor,
        image,
        questions,
        ground_truth_target,
        config_file="config.ini",
        is_targeted=False
    )
    
    rec.save("data/results/adv_donut/doa_optimized.jpg")

if __name__ == '__main__':
    print('Running Denial of Answer (DoA) attack')
    denial_of_answer_attack()

    print('Running multiple questions attack')
    multiple_questions_attack()

    print('Running single question attack')
    single_question_attack()