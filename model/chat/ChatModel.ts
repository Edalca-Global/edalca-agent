import mongoose, { Schema, Document } from "mongoose";

export type ChatType = "PRIVATE" | "GROUP" | "LOCATION";

export interface IChat extends Document {
  chatType: ChatType;
  name: string;
  sessions: mongoose.Types.ObjectId[];
  members: mongoose.Types.ObjectId[]; // User IDs
  groupId: mongoose.Types.ObjectId;
  locationId: mongoose.Types.ObjectId;
  isArchived: boolean,
  createdAt: Date;
  updatedAt: Date;
}

const ChatSchema = new mongoose.Schema(
  {
    name: { type: String, required: true },
    chatType: {
      type: String,
      enum: ["PRIVATE", "GROUP", "LOCATION"],
      required: true,
    },
    sessions: [{ type: mongoose.Schema.Types.ObjectId, ref: "ChatSession" }],
    members: [{ type: mongoose.Schema.Types.ObjectId, ref: "User" }],
    createdBy: { type: mongoose.Schema.Types.ObjectId, ref: "User" },
    groupId: { type: mongoose.Schema.Types.ObjectId, ref: "Group" },
    locationId: { type: mongoose.Schema.Types.ObjectId, ref: "Location" },
    isArchived: { type: Boolean, default: false },
    organizationId: {
      type: mongoose.Schema.Types.ObjectId,
      ref: "Organization",
    },
  },
  { timestamps: true }
);

const Chat = mongoose.models.Chat || mongoose.model<IChat>("Chat", ChatSchema);

export default Chat;
