use std::collections::HashMap;

use ndarray_rand::rand::SeedableRng;
use rand::rngs::SmallRng;

use ordered_float::OrderedFloat;

use linfa::prelude::*;
use linfa_trees::{DecisionTree, Result, SplitQuality};
use ndarray::{concatenate, s, ArrayBase, Axis, Dim, OwnedRepr, ViewRepr};

type VectorDataset = DatasetBase<
    ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>,
    ArrayBase<OwnedRepr<usize>, Dim<[usize; 1]>>,
>;

#[derive(Debug)]
struct Discriminator {
    feature: usize,
    threshold: f64,
}

#[derive(Debug)]
enum CustomDT {
    Branch {
        discriminator: Discriminator,
        less: Box<CustomDT>,
        more: Box<CustomDT>,
    },
    Leaf {
        label: usize,
    },
}

fn dataset_from_array(
    records: Vec<ArrayBase<ViewRepr<&f64>, Dim<[usize; 1]>>>,
    targets: Vec<usize>,
) -> VectorDataset {
    let records_views: Vec<_> = records
        .iter()
        .map(|arr| arr.view().insert_axis(Axis(0)))
        .collect();
    let records = concatenate(Axis(0), &records_views).unwrap();
    let targets: ArrayBase<OwnedRepr<usize>, Dim<[usize; 1]>> = ArrayBase::from_vec(targets);
    DatasetBase::new(records, targets)
}

impl CustomDT {
    fn predict(&self, record: &ArrayBase<ViewRepr<&f64>, Dim<[usize; 1]>>) -> usize {
        match self {
            CustomDT::Leaf { label } => *label,
            CustomDT::Branch {
                discriminator,
                less,
                more,
            } => {
                if record[discriminator.feature] <= discriminator.threshold {
                    less.predict(record)
                } else {
                    more.predict(record)
                }
            }
        }
    }
    fn predict_dataset(
        &self,
        dataset: &VectorDataset,
    ) -> ArrayBase<OwnedRepr<usize>, Dim<[usize; 1]>> {
        dataset
            .records()
            .outer_iter()
            .map(|record| self.predict(&record))
            .collect()
    }
    fn build(examples: &VectorDataset) -> CustomDT {
        // If all examples have the same label, return a leaf node with that label
        if examples
            .targets()
            .iter()
            .all(|&label| label == examples.targets()[0])
        {
            return CustomDT::Leaf {
                label: examples.targets()[0],
            };
        }
        let (_, discriminator) = (0..examples.records().dim().1)
            .map(|feature| advantage(examples, feature))
            .max_by(|(adv1, _), (adv2, _)| adv1.partial_cmp(adv2).unwrap())
            .unwrap();
        let (less, more): (Vec<_>, Vec<_>) = examples
            .records()
            .outer_iter()
            .zip(examples.targets().iter())
            .partition(|(record, _)| record[discriminator.feature] <= discriminator.threshold);
        let (less_records, less_targets): (Vec<_>, Vec<_>) = less.into_iter().unzip();
        let (more_records, more_targets): (Vec<_>, Vec<_>) = more.into_iter().unzip();
        CustomDT::Branch {
            discriminator,
            less: Box::new(CustomDT::build(&dataset_from_array(
                less_records,
                less_targets,
            ))),
            more: Box::new(CustomDT::build(&dataset_from_array(
                more_records,
                more_targets,
            ))),
        }
    }
}

fn bool_entropy(prob: f64) -> f64 {
    if prob == 0.0 || prob == 1.0 {
        0.0
    } else {
        -prob * prob.log2() - (1.0 - prob) * (1.0 - prob).log2()
    }
}

fn advantage(examples: &VectorDataset, feature: usize) -> (f64, Discriminator) {
    // Get the sorted values of the feature
    let mut values = examples
        .records()
        .slice(s![.., feature])
        .to_owned()
        .into_raw_vec();
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    // Get the amount of successes and examples for each value
    let mut cum_successes = vec![0; values.len()];
    let mut cum_examples = vec![0; values.len()];
    // Create a dictionary to map the values to their index
    let mut value_map = HashMap::new();
    for (i, value) in values.iter().enumerate() {
        value_map.insert(OrderedFloat(*value), i);
    }
    for (record, label) in examples
        .records()
        .outer_iter()
        .zip(examples.targets().iter())
    {
        let index = value_map.get(&OrderedFloat(record[feature])).unwrap();
        if *label >= 1 {
            cum_successes[*index] += 1;
        }
        cum_examples[*index] = 1;
    }
    // Accumulate the successes and examples
    for i in 1..values.len() {
        cum_successes[i] += cum_successes[i - 1];
        cum_examples[i] += cum_examples[i - 1];
    }
    // Calculate the minimum reminder
    let mut min_reminder = 1.;
    let mut best_split = 0;
    for i in 0..values.len() - 1 {
        let reminder = cum_examples[i] as f64 / examples.records.len() as f64
            * bool_entropy(cum_successes[i] as f64 / cum_examples[i] as f64)
            + (examples.targets.len() - cum_examples[i]) as f64 / examples.records.len() as f64
                * bool_entropy(
                    (cum_successes[values.len() - 1] - cum_successes[i]) as f64
                        / (examples.targets.len() - cum_examples[i]) as f64,
                );
        if reminder < min_reminder {
            min_reminder = reminder;
            best_split = i;
        }
    }
    (
        bool_entropy(cum_successes[values.len() - 1] as f64 / examples.targets.len() as f64)
            - min_reminder,
        Discriminator {
            feature,
            threshold: values[best_split],
        },
    )
}
fn main() -> Result<()> {
    // let mut rng = SmallRng::seed_from_u64(42);
    let mut rng = SmallRng::from_entropy();
    // Remove the data if target is 2
    let dataset = linfa_datasets::winequality();
    let (records, targets): (Vec<_>, Vec<_>) = dataset
        .records()
        .outer_iter()
        .zip(dataset.targets().iter())
        .filter(|(_, &target)| target != 2)
        .unzip();
    let dataset = dataset_from_array(records, targets);
    let (train, test) = dataset.shuffle(&mut rng).split_with_ratio(0.5);
    println!("Training model with entropy criterion ...");
    let entropy_model = DecisionTree::params()
        .split_quality(SplitQuality::Entropy)
        .max_depth(Some(100))
        // .min_weight_split(10.0)
        // .min_weight_leaf(10.0)
        .fit(&train)?;
    println!("Training custom model ...");
    let model = CustomDT::build(&train);
    println!("Predicting with custom model ...");
    let pred_y = model.predict_dataset(&test);
    let cm = pred_y.confusion_matrix(&test)?;
    println!("{:?}", cm);
    println!(
        "Test accuracy with custom criterion: {:.2}%",
        100.0 * cm.accuracy()
    );
    let pred_train = model.predict_dataset(&train);
    let cm = pred_train.confusion_matrix(&train)?;
    println!("{:?}", cm);
    println!(
        "Train accuracy with custom DT: {:.2}%",
        100.0 * cm.accuracy()
    );
    println!("Predicting with entropy model ...");
    let entropy_pred_y = entropy_model.predict(&test);
    let cm = entropy_pred_y.confusion_matrix(&test)?;

    println!("{:?}", cm);

    println!(
        "Test accuracy with Entropy criterion: {:.2}%",
        100.0 * cm.accuracy()
    );
    println!("{:?}", model);
    Ok(())
}
