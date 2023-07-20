use std::fs::{read_to_string};
use rand::Rng;

struct Question {
    question: String,
    answer: String,
    correctly_guessed: bool,
}
impl Question {
    pub fn new(question: String, answer: String) -> Question {
        Question{question, answer, correctly_guessed: false}
    }
    //test this function please
    pub fn get_questions_from_file(path_to_file: String) -> Vec<Question> {
        let file_content = read_to_string(path_to_file).expect("error in reading the file"); //change this to a match statement
        let questions_answers_file_content = file_content.split("\n").map(|x|x.to_string()).collect::<Vec<String>>();
        if questions_answers_file_content.len() % 2 != 0 {
            panic!("question with missing answer")
        }
        else {
            let mut ret_vec = Vec::new();
            let (mut i, mut j) = (0, 1);
            while i < questions_answers_file_content.len() / 2 {
                let question = Question::new(
                    questions_answers_file_content.get(i).unwrap().clone(),
                     questions_answers_file_content.get(j).unwrap().clone()
                );
                ret_vec.push(question);
                i += 2;
                j += 2;
            }
            ret_vec
        }
    }
    /// ATTENTION
    /// function returns true and an index from the correctly guessed questions if the rng is 5
    /// otherwise it returns false and an index
    pub fn get_random_question_index(correctly_guessed_questions: Vec<Question>, not_guessed_questions: Vec<Question>) -> (bool, usize) {
        let rng = rand::thread_rng().gen_range(1..=5);
        if rng == 5 && correctly_guessed_questions.len() > 0{
            return (true, rand::thread_rng().gen_range(0..correctly_guessed_questions.len()));
        }
        return (false, rand::thread_rng().gen_range(0..not_guessed_questions.len()));
    }
}