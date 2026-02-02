use dashtable::table::DashTable;

fn main() {
    let mut table = DashTable::<u64, String>::new();

    table.insert(1, "hello".to_string());

    assert!(table.contains_key(&1));

    let value = table.get(&1);

    println!("Value: {:?}", value);
}
