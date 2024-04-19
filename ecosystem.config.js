module.exports = {
    apps: [{
        name: "my_uv_app",
        script: "uvicorn",
        args: ["main:app", "--reload"],
        instances: "max",
        exec_mode: "cluster"
    }]
}
