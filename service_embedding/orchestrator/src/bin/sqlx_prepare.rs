use std::env;
use std::process::Command;
use dotenv::dotenv;

fn main() {
    dotenv().ok();

    let host = env::var("POSTGRES_HOST").unwrap_or("localhost".into());
    let port = env::var("POSTGRES_PORT").unwrap_or("5432".into());
    let user = env::var("POSTGRES_USER").unwrap_or("postgres".into());
    let pass = env::var("POSTGRES_PASSWORD").unwrap_or("".into());
    let db   = env::var("POSTGRES_DB").unwrap_or("postgres".into());

    let db_url = format!("postgres://{}:{}@{}:{}/{}", user, pass, host, port, db);

    let status = Command::new("cargo")
        .args(&["sqlx", "prepare"])
        .env("DATABASE_URL", db_url)
        .status()
        .expect("failed to run cargo sqlx prepare");

    std::process::exit(status.code().unwrap_or(1));
}
