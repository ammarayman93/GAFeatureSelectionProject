const mongoose = require('mongoose')
const Schema=mongoose.Schema
const schema = mongoose.Schema(
    {
        type: String,
        description: String,
        notes: String,
        file: String,
        status: String,
        insertD: String,
        updateD: String,
        user: {
            type: Schema.Types.ObjectId,
            ref: "User"
        },
        customer: {
            type: Schema.Types.ObjectId,
            ref: "Customer"
        }

    }
);
module.exports = mongoose.model("Order", schema);